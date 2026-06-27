package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/xai"
	"github.com/gin-gonic/gin"
)

func (s *OpenAIGatewayService) forwardGrokImages(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	parsed *OpenAIImagesRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	startTime := time.Now()
	if account.Type != AccountTypeOAuth && account.Type != AccountTypeAPIKey {
		return nil, fmt.Errorf("grok account type %s is not supported by image forwarding", account.Type)
	}
	if parsed.Endpoint != openAIImagesGenerationsEndpoint && parsed.Endpoint != openAIImagesEditsEndpoint {
		return nil, fmt.Errorf("grok image endpoint %s is not supported", parsed.Endpoint)
	}
	if parsed.Multipart {
		return nil, fmt.Errorf("grok images endpoint requires a JSON request body")
	}
	if parsed.Stream {
		return nil, fmt.Errorf("grok images endpoint does not support stream responses")
	}
	if parsed.IsEdits() {
		if parsed.HasMask {
			return nil, fmt.Errorf("grok image edits endpoint does not support mask inputs")
		}
		if len(parsed.InputImageURLs) == 0 {
			return nil, fmt.Errorf("grok image edits endpoint requires at least one image input")
		}
		if len(parsed.InputImageURLs) > 3 {
			return nil, fmt.Errorf("grok image edits endpoint supports at most 3 image inputs")
		}
	}

	requestModel := strings.TrimSpace(parsed.Model)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		requestModel = mapped
	}
	if requestModel == "" {
		requestModel = "grok-imagine-image"
	}
	if !xai.IsImageGenerationModel(requestModel) {
		return nil, fmt.Errorf("grok images endpoint requires a grok image model, got %q", requestModel)
	}
	upstreamModel := account.GetMappedModel(requestModel)
	if !xai.IsImageGenerationModel(upstreamModel) {
		return nil, fmt.Errorf("grok images endpoint requires a grok image upstream model, got %q", upstreamModel)
	}

	forwardBody, forwardContentType, err := rewriteGrokImagesBody(body, parsed.ContentType, parsed, upstreamModel)
	if err != nil {
		return nil, err
	}

	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		return nil, err
	}

	upstreamCtx, releaseUpstreamCtx := detachUpstreamContext(ctx)
	defer releaseUpstreamCtx()
	upstreamReq, err := buildGrokImagesRequest(upstreamCtx, c, account, parsed.Endpoint, forwardBody, forwardContentType, token)
	if err != nil {
		return nil, err
	}

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		return nil, s.handleOpenAIUpstreamTransportError(ctx, c, account, err, false)
	}
	if resp, err = s.retryGrokOAuthUnauthorized(ctx, account, resp, proxyURL, func(token string) (*http.Request, error) {
		return buildGrokImagesRequest(upstreamCtx, c, account, parsed.Endpoint, forwardBody, forwardContentType, token)
	}); err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody := s.readUpstreamErrorBody(resp)
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		s.updateGrokUsageSnapshot(ctx, account.ID, xai.ParseQuotaHeaders(resp.Header, resp.StatusCode))
		upstreamMsg := sanitizeUpstreamErrorMessage(extractUpstreamErrorMessage(respBody))
		if upstreamMsg == "" {
			upstreamMsg = fmt.Sprintf("xAI image upstream returned status %d", resp.StatusCode)
		}
		appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
			Platform:           account.Platform,
			AccountID:          account.ID,
			AccountName:        account.Name,
			UpstreamStatusCode: resp.StatusCode,
			UpstreamRequestID:  firstNonEmpty(resp.Header.Get("x-request-id"), resp.Header.Get("xai-request-id")),
			UpstreamURL:        safeUpstreamURL(upstreamReq.URL.String()),
			Kind:               "failover",
			Message:            upstreamMsg,
		})
		s.handleGrokAccountUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)
		if s.shouldFailoverUpstreamError(resp.StatusCode) {
			return nil, &UpstreamFailoverError{
				StatusCode:             resp.StatusCode,
				ResponseBody:           respBody,
				RetryableOnSameAccount: account.IsPoolMode() && account.IsPoolModeRetryableStatus(resp.StatusCode),
			}
		}
		return s.handleOpenAIImagesErrorResponse(ctx, resp, c, account, upstreamModel)
	}

	s.updateGrokUsageSnapshot(ctx, account.ID, xai.ParseQuotaHeaders(resp.Header, resp.StatusCode))

	imageCount := parsed.N
	var usage OpenAIUsage
	var firstTokenMs *int
	var imageOutputSizes []string
	if parsed.Stream && isEventStreamResponse(resp.Header) {
		streamUsage, streamCount, streamSizes, ttft, err := s.handleOpenAIImagesStreamingResponse(resp, c, startTime)
		if err != nil {
			return nil, err
		}
		usage = streamUsage
		if streamCount > 0 {
			imageCount = streamCount
		}
		firstTokenMs = ttft
		imageOutputSizes = streamSizes
	} else {
		nonStreamUsage, nonStreamCount, nonStreamSizes, err := s.handleOpenAIImagesNonStreamingResponse(resp, c)
		if err != nil {
			return nil, err
		}
		usage = nonStreamUsage
		if nonStreamCount > 0 {
			imageCount = nonStreamCount
		}
		imageOutputSizes = nonStreamSizes
	}

	return &OpenAIForwardResult{
		RequestID:        firstNonEmpty(resp.Header.Get("x-request-id"), resp.Header.Get("xai-request-id")),
		Usage:            usage,
		Model:            requestModel,
		UpstreamModel:    upstreamModel,
		Stream:           parsed.Stream,
		ResponseHeaders:  resp.Header.Clone(),
		Duration:         time.Since(startTime),
		FirstTokenMs:     firstTokenMs,
		ImageCount:       imageCount,
		ImageSize:        parsed.SizeTier,
		ImageInputSize:   parsed.Size,
		ImageOutputSizes: imageOutputSizes,
	}, nil
}

func rewriteGrokImagesBody(body []byte, contentType string, parsed *OpenAIImagesRequest, upstreamModel string) ([]byte, string, error) {
	if parsed == nil || !parsed.IsEdits() {
		return rewriteOpenAIImagesModel(body, contentType, upstreamModel)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, "", fmt.Errorf("parse grok image edits body: %w", err)
	}
	payload["model"] = upstreamModel
	if err := normalizeGrokImageEditsPayload(payload, parsed.InputImageURLs); err != nil {
		return nil, "", err
	}
	rewritten, err := json.Marshal(payload)
	if err != nil {
		return nil, "", fmt.Errorf("rewrite grok image edits body: %w", err)
	}
	return rewritten, firstNonEmpty(strings.TrimSpace(contentType), "application/json"), nil
}

func normalizeGrokImageEditsPayload(payload map[string]any, imageURLs []string) error {
	if payload == nil {
		return fmt.Errorf("grok image edits body is required")
	}

	if rawImage, ok := payload["image"]; ok {
		image, err := normalizeGrokImageReference(rawImage)
		if err != nil {
			return fmt.Errorf("invalid image input: %w", err)
		}
		payload["image"] = image
		delete(payload, "images")
		delete(payload, "image_url")
		delete(payload, "image_urls")
		return nil
	}

	refs := make([]map[string]any, 0, len(imageURLs))
	for _, imageURL := range imageURLs {
		imageURL = strings.TrimSpace(imageURL)
		if imageURL == "" {
			continue
		}
		refs = append(refs, grokImageReference(imageURL))
	}
	if len(refs) == 0 {
		return fmt.Errorf("grok image edits endpoint requires at least one image input")
	}
	delete(payload, "images")
	delete(payload, "image_url")
	delete(payload, "image_urls")
	if len(refs) == 1 {
		payload["image"] = refs[0]
		return nil
	}
	images := make([]any, 0, len(refs))
	for _, ref := range refs {
		images = append(images, ref)
	}
	payload["images"] = images
	return nil
}

func normalizeGrokImageReference(raw any) (map[string]any, error) {
	switch value := raw.(type) {
	case string:
		value = strings.TrimSpace(value)
		if value == "" {
			return nil, fmt.Errorf("url is required")
		}
		return grokImageReference(value), nil
	case map[string]any:
		url := ""
		if v, ok := value["url"].(string); ok {
			url = strings.TrimSpace(v)
		}
		if url == "" {
			if v, ok := value["image_url"].(string); ok {
				url = strings.TrimSpace(v)
			}
		}
		if url == "" {
			return nil, fmt.Errorf("url is required")
		}
		value["url"] = url
		delete(value, "image_url")
		if _, ok := value["type"].(string); !ok || strings.TrimSpace(fmt.Sprint(value["type"])) == "" {
			value["type"] = "image_url"
		}
		return value, nil
	default:
		return nil, fmt.Errorf("unsupported image input type")
	}
}

func grokImageReference(url string) map[string]any {
	return map[string]any{
		"url":  strings.TrimSpace(url),
		"type": "image_url",
	}
}

func buildGrokImagesRequest(ctx context.Context, c *gin.Context, account *Account, endpoint string, body []byte, contentType string, token string) (*http.Request, error) {
	var targetURL string
	var err error
	switch endpoint {
	case openAIImagesGenerationsEndpoint:
		targetURL, err = xai.BuildImagesGenerationsURL(account.GetGrokBaseURL())
	case openAIImagesEditsEndpoint:
		targetURL, err = xai.BuildImagesEditsURL(account.GetGrokBaseURL())
	default:
		return nil, fmt.Errorf("grok image endpoint %s is not supported", endpoint)
	}
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", firstNonEmpty(strings.TrimSpace(contentType), "application/json"))
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("User-Agent", "sub2api-grok-images/1.0")
	if c != nil {
		if v := c.GetHeader("OpenAI-Beta"); strings.TrimSpace(v) != "" {
			req.Header.Set("OpenAI-Beta", v)
		}
	}
	return req, nil
}

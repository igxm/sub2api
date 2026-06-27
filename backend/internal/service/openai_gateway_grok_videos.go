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
	"github.com/tidwall/gjson"
)

func (s *OpenAIGatewayService) forwardGrokVideos(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	parsed *OpenAIVideosRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	startTime := time.Now()
	if account.Type != AccountTypeOAuth && account.Type != AccountTypeAPIKey {
		return nil, fmt.Errorf("grok account type %s is not supported by video forwarding", account.Type)
	}

	requestModel := strings.TrimSpace(parsed.Model)
	upstreamModel := requestModel
	forwardBody := body
	forwardContentType := "application/json"
	if !parsed.IsResult() {
		if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
			upstreamModel = mapped
		}
		if upstreamModel == "" {
			return nil, fmt.Errorf("grok videos endpoint requires a model")
		}
		if !isGrokVideoModel(upstreamModel) {
			return nil, fmt.Errorf("grok videos endpoint requires a grok video model, got %q", upstreamModel)
		}
		var err error
		forwardBody, forwardContentType, err = rewriteGrokVideosBody(body, upstreamModel)
		if err != nil {
			return nil, err
		}
	}

	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		return nil, err
	}

	upstreamCtx, releaseUpstreamCtx := detachUpstreamContext(ctx)
	defer releaseUpstreamCtx()
	upstreamReq, err := buildGrokVideosRequest(upstreamCtx, account, parsed, forwardBody, forwardContentType, token)
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
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody := s.readUpstreamErrorBody(resp)
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		s.updateGrokUsageSnapshot(ctx, account.ID, xai.ParseQuotaHeaders(resp.Header, resp.StatusCode))
		upstreamMsg := sanitizeUpstreamErrorMessage(extractUpstreamErrorMessage(respBody))
		if upstreamMsg == "" {
			upstreamMsg = fmt.Sprintf("xAI video upstream returned status %d", resp.StatusCode)
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
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read xAI video response: %w", err)
	}
	for key, values := range resp.Header {
		for _, value := range values {
			c.Writer.Header().Add(key, value)
		}
	}
	c.Status(resp.StatusCode)
	_, _ = c.Writer.Write(responseBody)

	requestID := firstNonEmpty(resp.Header.Get("x-request-id"), resp.Header.Get("xai-request-id"))
	if parsed.IsResult() {
		requestID = firstNonEmpty(parsed.RequestID, requestID)
	} else if upstreamRequestID := strings.TrimSpace(gjson.GetBytes(responseBody, "request_id").String()); upstreamRequestID != "" {
		requestID = firstNonEmpty(resp.Header.Get("x-request-id"), resp.Header.Get("xai-request-id"), upstreamRequestID)
	}

	return &OpenAIForwardResult{
		RequestID:       requestID,
		Model:           requestModel,
		UpstreamModel:   upstreamModel,
		ResponseHeaders: resp.Header.Clone(),
		Duration:        time.Since(startTime),
	}, nil
}

func buildGrokVideosRequest(ctx context.Context, account *Account, parsed *OpenAIVideosRequest, body []byte, contentType string, token string) (*http.Request, error) {
	targetURL, err := grokVideosTargetURL(account.GetGrokBaseURL(), parsed)
	if err != nil {
		return nil, err
	}
	method := http.MethodPost
	var reader io.Reader = bytes.NewReader(body)
	if parsed.IsResult() {
		method = http.MethodGet
		reader = nil
	}
	req, err := http.NewRequestWithContext(ctx, method, targetURL, reader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/json")
	if !parsed.IsResult() {
		req.Header.Set("Content-Type", firstNonEmpty(strings.TrimSpace(contentType), "application/json"))
	}
	req.Header.Set("User-Agent", "sub2api-grok-videos/1.0")
	return req, nil
}

func rewriteGrokVideosBody(body []byte, upstreamModel string) ([]byte, string, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, "", fmt.Errorf("parse grok video body: %w", err)
	}
	payload["model"] = upstreamModel
	normalizeGrokVideoSDKFields(payload)
	rewritten, err := json.Marshal(payload)
	if err != nil {
		return nil, "", fmt.Errorf("rewrite grok video body: %w", err)
	}
	return rewritten, "application/json", nil
}

func normalizeGrokVideoSDKFields(payload map[string]any) {
	if payload == nil {
		return
	}
	if _, exists := payload["image"]; !exists {
		if imageURL := strings.TrimSpace(grokVideoStringFromAny(payload["image_url"])); imageURL != "" {
			payload["image"] = map[string]any{"url": imageURL}
		}
	}
	delete(payload, "image_url")

	if _, exists := payload["reference_images"]; !exists {
		if refs := imageRefsFromAny(payload["reference_image_urls"]); len(refs) > 0 {
			payload["reference_images"] = refs
		}
	}
	delete(payload, "reference_image_urls")

	if _, exists := payload["video"]; !exists {
		if videoURL := strings.TrimSpace(grokVideoStringFromAny(payload["video_url"])); videoURL != "" {
			payload["video"] = map[string]any{"url": videoURL}
		}
	}
	delete(payload, "video_url")
}

func imageRefsFromAny(raw any) []map[string]any {
	values, ok := raw.([]any)
	if !ok {
		return nil
	}
	refs := make([]map[string]any, 0, len(values))
	for _, value := range values {
		url := strings.TrimSpace(grokVideoStringFromAny(value))
		if url == "" {
			continue
		}
		refs = append(refs, map[string]any{"url": url})
	}
	return refs
}

func grokVideoStringFromAny(raw any) string {
	value, _ := raw.(string)
	return value
}

func grokVideosTargetURL(baseURL string, parsed *OpenAIVideosRequest) (string, error) {
	switch parsed.Endpoint {
	case openAIVideosGenerationsEndpoint:
		return xai.BuildVideosGenerationsURL(baseURL)
	case openAIVideosEditsEndpoint:
		return xai.BuildVideosEditsURL(baseURL)
	case openAIVideosExtensionsEndpoint:
		return xai.BuildVideosExtensionsURL(baseURL)
	case openAIVideosResultEndpoint:
		return xai.BuildVideoResultURL(baseURL, parsed.RequestID)
	default:
		return "", fmt.Errorf("grok video endpoint %s is not supported", parsed.Endpoint)
	}
}

func isGrokVideoModel(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	return model == "grok-imagine-video" || strings.HasPrefix(model, "grok-imagine-video-")
}

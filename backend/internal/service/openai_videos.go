package service

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

const (
	openAIVideosGenerationsEndpoint = "/v1/videos/generations"
	openAIVideosEditsEndpoint       = "/v1/videos/edits"
	openAIVideosExtensionsEndpoint  = "/v1/videos/extensions"
	openAIVideosResultEndpoint      = "/v1/videos"
)

type OpenAIVideosRequest struct {
	Endpoint        string
	Method          string
	Model           string
	RequestID       string
	Body            []byte
	DurationSeconds int
}

func (r *OpenAIVideosRequest) IsResult() bool {
	return r != nil && r.Endpoint == openAIVideosResultEndpoint
}

func (s *OpenAIGatewayService) ParseOpenAIVideosRequest(c *gin.Context, body []byte) (*OpenAIVideosRequest, error) {
	if c == nil || c.Request == nil {
		return nil, fmt.Errorf("missing request context")
	}
	endpoint, requestID := normalizeOpenAIVideosEndpointPath(c.Request.URL.Path)
	if endpoint == "" {
		return nil, fmt.Errorf("unsupported videos endpoint")
	}

	req := &OpenAIVideosRequest{
		Endpoint:  endpoint,
		Method:    c.Request.Method,
		RequestID: requestID,
		Body:      body,
	}
	if req.IsResult() {
		if c.Request.Method != "GET" {
			return nil, fmt.Errorf("video result endpoint requires GET")
		}
		if requestID == "" {
			return nil, fmt.Errorf("video request id is required")
		}
		return req, nil
	}
	if c.Request.Method != "POST" {
		return nil, fmt.Errorf("video endpoint requires POST")
	}
	if len(body) == 0 {
		return nil, fmt.Errorf("request body is empty")
	}
	if !gjson.ValidBytes(body) {
		return nil, fmt.Errorf("failed to parse request body")
	}
	if modelResult := gjson.GetBytes(body, "model"); modelResult.Exists() {
		req.Model = strings.TrimSpace(modelResult.String())
	}
	if strings.TrimSpace(req.Model) == "" {
		return nil, fmt.Errorf("videos endpoint requires a model")
	}
	req.DurationSeconds = parseOpenAIVideoDurationSeconds(body)
	return req, nil
}

func parseOpenAIVideoDurationSeconds(body []byte) int {
	const defaultVideoDurationSeconds = 5
	if len(body) == 0 || !gjson.ValidBytes(body) {
		return defaultVideoDurationSeconds
	}
	duration := gjson.GetBytes(body, "duration")
	if !duration.Exists() {
		return defaultVideoDurationSeconds
	}
	switch duration.Type {
	case gjson.Number:
		if duration.Int() > 0 {
			return int(duration.Int())
		}
	case gjson.String:
		value := strings.TrimSpace(duration.String())
		value = strings.TrimSuffix(value, "s")
		if parsed, err := strconv.Atoi(value); err == nil && parsed > 0 {
			return parsed
		}
	}
	return defaultVideoDurationSeconds
}

func normalizeOpenAIVideosEndpointPath(path string) (string, string) {
	trimmed := strings.Trim(strings.TrimSpace(path), "/")
	switch {
	case trimmed == "v1/videos/generations" || trimmed == "videos/generations":
		return openAIVideosGenerationsEndpoint, ""
	case trimmed == "v1/videos/edits" || trimmed == "videos/edits":
		return openAIVideosEditsEndpoint, ""
	case trimmed == "v1/videos/extensions" || trimmed == "videos/extensions":
		return openAIVideosExtensionsEndpoint, ""
	case strings.HasPrefix(trimmed, "v1/videos/"):
		return openAIVideosResultEndpoint, strings.TrimSpace(strings.TrimPrefix(trimmed, "v1/videos/"))
	case strings.HasPrefix(trimmed, "videos/"):
		return openAIVideosResultEndpoint, strings.TrimSpace(strings.TrimPrefix(trimmed, "videos/"))
	default:
		return "", ""
	}
}

func VideoGenerationPlatformForModel(model string) string {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "grok-imagine-video") {
		return PlatformGrok
	}
	return PlatformOpenAI
}

func (s *OpenAIGatewayService) ForwardVideos(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	parsed *OpenAIVideosRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed videos request is required")
	}
	if account.Platform == PlatformGrok {
		return s.forwardGrokVideos(ctx, c, account, body, parsed, channelMappedModel)
	}
	return nil, fmt.Errorf("videos endpoint is not supported for platform %s", account.Platform)
}

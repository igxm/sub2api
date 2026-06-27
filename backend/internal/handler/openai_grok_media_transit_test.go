//go:build unit

package handler

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/xai"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type grokMediaTransitUpstream struct {
	service.HTTPUpstream
	mu       sync.Mutex
	requests []*http.Request
	bodies   [][]byte
}

func (u *grokMediaTransitUpstream) Do(req *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	body := []byte(nil)
	if req != nil && req.Body != nil {
		body, _ = io.ReadAll(req.Body)
		_ = req.Body.Close()
		req.Body = io.NopCloser(bytes.NewReader(body))
	}
	u.mu.Lock()
	u.requests = append(u.requests, req)
	u.bodies = append(u.bodies, append([]byte(nil), body...))
	u.mu.Unlock()

	switch req.URL.Path {
	case "/v1/images/generations":
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}, "Xai-Request-Id": []string{"xai-image-handler"}},
			Body:       io.NopCloser(strings.NewReader(`{"created":1710000000,"data":[{"b64_json":"iVBORw0KGgo="}]}`)),
		}, nil
	case "/v1/videos/generations":
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}, "Xai-Request-Id": []string{"xai-video-handler"}},
			Body:       io.NopCloser(strings.NewReader(`{"request_id":"video_req_handler_123","status":"queued"}`)),
		}, nil
	default:
		return &http.Response{
			StatusCode: http.StatusNotFound,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"unexpected path"}}`)),
		}, nil
	}
}

func (u *grokMediaTransitUpstream) recorded() ([]*http.Request, [][]byte) {
	u.mu.Lock()
	defer u.mu.Unlock()
	return append([]*http.Request(nil), u.requests...), append([][]byte(nil), u.bodies...)
}

func TestOpenAIGatewayHandlerGrokMediaTransit_ImagesAndVideosGenerateThroughGateway(t *testing.T) {
	t.Setenv(xai.EnvAllowUnsafeURLOverrides, "true")
	gin.SetMode(gin.TestMode)
	groupID := int64(6227)
	accounts := []service.Account{{
		ID:          72,
		Name:        "grok-media",
		Platform:    service.PlatformGrok,
		Type:        service.AccountTypeAPIKey,
		Status:      service.StatusActive,
		Schedulable: true,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "xai-key",
			"base_url": "https://xai.test/v1",
			"model_mapping": map[string]any{
				"grok-imagine-image": "grok-imagine-image",
				"grok-imagine-video": "grok-imagine-video",
			},
		},
	}}
	accountRepo := openAIImagesFailoverAccountRepo{accounts: accounts}
	upstream := &grokMediaTransitUpstream{}
	cfg := &config.Config{RunMode: config.RunModeSimple}
	gatewayService := service.NewOpenAIGatewayService(
		accountRepo,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		cfg,
		nil,
		nil,
		nil,
		nil,
		nil,
		upstream,
		service.NewDeferredService(accountRepo, nil, time.Hour),
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
	)
	billingService := service.NewBillingCacheService(nil, nil, nil, nil, nil, nil, cfg, nil)
	t.Cleanup(billingService.Stop)
	handler := NewOpenAIGatewayHandler(
		gatewayService,
		service.NewConcurrencyService(nil),
		billingService,
		service.NewAPIKeyService(nil, nil, nil, nil, nil, nil, cfg),
		nil,
		nil,
		nil,
		nil,
		cfg,
	)

	imageBody := []byte(`{"model":"grok-imagine-image","prompt":"draw a cat","response_format":"b64_json"}`)
	imageRec, imageCtx := newGrokMediaTransitContext(t, http.MethodPost, "/v1/images/generations", imageBody, groupID)
	handler.Images(imageCtx)
	require.Equal(t, http.StatusOK, imageRec.Code)
	require.Equal(t, "iVBORw0KGgo=", gjson.GetBytes(imageRec.Body.Bytes(), "data.0.b64_json").String())

	videoBody := []byte(`{"model":"grok-imagine-video","prompt":"a red paper boat","duration":1}`)
	videoRec, videoCtx := newGrokMediaTransitContext(t, http.MethodPost, "/v1/videos/generations", videoBody, groupID)
	handler.Videos(videoCtx)
	require.Equal(t, http.StatusOK, videoRec.Code)
	require.Equal(t, "video_req_handler_123", gjson.GetBytes(videoRec.Body.Bytes(), "request_id").String())

	requests, bodies := upstream.recorded()
	require.Len(t, requests, 2)
	require.Equal(t, "https://xai.test/v1/images/generations", requests[0].URL.String())
	require.Equal(t, "Bearer xai-key", requests[0].Header.Get("Authorization"))
	require.Equal(t, "grok-imagine-image", gjson.GetBytes(bodies[0], "model").String())
	require.Equal(t, "https://xai.test/v1/videos/generations", requests[1].URL.String())
	require.Equal(t, "Bearer xai-key", requests[1].Header.Get("Authorization"))
	require.Equal(t, "grok-imagine-video", gjson.GetBytes(bodies[1], "model").String())
}

func TestOpenAIGatewayHandlerGrokMediaTransit_VideoGenerationRequiresGroupPermission(t *testing.T) {
	t.Setenv(xai.EnvAllowUnsafeURLOverrides, "true")
	gin.SetMode(gin.TestMode)
	groupID := int64(6228)
	accounts := []service.Account{{
		ID:          73,
		Name:        "grok-video",
		Platform:    service.PlatformGrok,
		Type:        service.AccountTypeAPIKey,
		Status:      service.StatusActive,
		Schedulable: true,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "xai-key",
			"base_url": "https://xai.test/v1",
			"model_mapping": map[string]any{
				"grok-imagine-video": "grok-imagine-video",
			},
		},
	}}
	accountRepo := openAIImagesFailoverAccountRepo{accounts: accounts}
	upstream := &grokMediaTransitUpstream{}
	cfg := &config.Config{RunMode: config.RunModeSimple}
	gatewayService := service.NewOpenAIGatewayService(
		accountRepo,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		cfg,
		nil,
		nil,
		nil,
		nil,
		nil,
		upstream,
		service.NewDeferredService(accountRepo, nil, time.Hour),
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
	)
	billingService := service.NewBillingCacheService(nil, nil, nil, nil, nil, nil, cfg, nil)
	t.Cleanup(billingService.Stop)
	handler := NewOpenAIGatewayHandler(
		gatewayService,
		service.NewConcurrencyService(nil),
		billingService,
		service.NewAPIKeyService(nil, nil, nil, nil, nil, nil, cfg),
		nil,
		nil,
		nil,
		nil,
		cfg,
	)

	videoBody := []byte(`{"model":"grok-imagine-video","prompt":"a red paper boat","duration":1}`)
	videoRec, videoCtx := newGrokMediaTransitContext(t, http.MethodPost, "/v1/videos/generations", videoBody, groupID)
	videoCtx.MustGet(string(middleware2.ContextKeyAPIKey)).(*service.APIKey).Group.AllowVideoGeneration = false

	handler.Videos(videoCtx)

	require.Equal(t, http.StatusForbidden, videoRec.Code)
	require.Equal(t, "permission_error", gjson.GetBytes(videoRec.Body.Bytes(), "error.type").String())
	require.Contains(t, gjson.GetBytes(videoRec.Body.Bytes(), "error.message").String(), "Video generation is not enabled")
	requests, _ := upstream.recorded()
	require.Empty(t, requests)
}

func newGrokMediaTransitContext(t *testing.T, method string, path string, body []byte, groupID int64) (*httptest.ResponseRecorder, *gin.Context) {
	t.Helper()
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(method, path, bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Set(string(middleware2.ContextKeyAPIKey), &service.APIKey{
		ID:      99,
		GroupID: &groupID,
		Group: &service.Group{
			ID:                   groupID,
			Platform:             service.PlatformGrok,
			AllowImageGeneration: true,
			AllowVideoGeneration: true,
		},
		User: &service.User{ID: 100},
	})
	c.Set(string(middleware2.ContextKeyUser), middleware2.AuthSubject{UserID: 100, Concurrency: 0})
	c.Request = c.Request.WithContext(context.Background())
	return rec, c
}

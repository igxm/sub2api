//go:build unit

package handler

import (
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func TestShouldRecordOpenAIVideosUsageSkipsResultPolls(t *testing.T) {
	require.False(t, shouldRecordOpenAIVideosUsage(nil))
	require.False(t, shouldRecordOpenAIVideosUsage(&service.OpenAIVideosRequest{
		Endpoint:  "/v1/videos",
		Method:    "GET",
		RequestID: "video_req_123",
	}))
	require.True(t, shouldRecordOpenAIVideosUsage(&service.OpenAIVideosRequest{
		Endpoint: "/v1/videos/generations",
		Method:   "POST",
		Model:    "grok-imagine-video",
	}))
}

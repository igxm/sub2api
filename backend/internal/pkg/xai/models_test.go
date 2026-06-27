package xai

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDefaultModelIDsIncludeImagineVideoFallbacks(t *testing.T) {
	ids := DefaultModelIDs()

	require.Contains(t, ids, "grok-imagine-image")
	require.Contains(t, ids, "grok-imagine-image-quality")
	require.Contains(t, ids, "grok-imagine-video")
	require.Contains(t, ids, "grok-imagine-video-1.5")
}

func TestDefaultModelMappingIncludesImagineVideoFallbacks(t *testing.T) {
	mapping := DefaultModelMapping()

	require.Equal(t, "grok-imagine-video", mapping["grok-imagine-video"])
	require.Equal(t, "grok-imagine-video-1.5", mapping["grok-imagine-video-1.5"])
}

package service

import (
	"context"
	"fmt"
	"net/http"
)

func (s *OpenAIGatewayService) retryGrokOAuthUnauthorized(
	ctx context.Context,
	account *Account,
	resp *http.Response,
	proxyURL string,
	buildRequest func(token string) (*http.Request, error),
) (*http.Response, error) {
	if resp == nil || resp.StatusCode != http.StatusUnauthorized {
		return resp, nil
	}
	if s == nil || account == nil || account.Platform != PlatformGrok || account.Type != AccountTypeOAuth || s.grokTokenProvider == nil {
		return resp, nil
	}
	if resp.Body != nil {
		_ = resp.Body.Close()
	}

	token, err := s.grokTokenProvider.ForceRefreshAccessToken(ctx, account)
	if err != nil {
		return nil, fmt.Errorf("refresh grok access token after upstream 401: %w", err)
	}
	retryReq, err := buildRequest(token)
	if err != nil {
		return nil, fmt.Errorf("build grok retry request after upstream 401: %w", err)
	}
	retryResp, err := s.httpUpstream.Do(retryReq, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return nil, s.handleOpenAIUpstreamTransportError(ctx, nil, account, err, false)
	}
	return retryResp, nil
}

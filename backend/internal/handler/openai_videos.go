package handler

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"time"

	pkghttputil "github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"github.com/Wei-Shaw/sub2api/internal/pkg/ip"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// Videos handles OpenAI-compatible async video endpoints.
// POST /v1/videos/generations
// POST /v1/videos/edits
// POST /v1/videos/extensions
// GET  /v1/videos/:request_id
func (h *OpenAIGatewayHandler) Videos(c *gin.Context) {
	streamStarted := false
	defer h.recoverResponsesPanic(c, &streamStarted)

	requestStart := time.Now()

	apiKey, ok := middleware2.GetAPIKeyFromContext(c)
	if !ok {
		h.errorResponse(c, http.StatusUnauthorized, "authentication_error", "Invalid API key")
		return
	}

	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		h.errorResponse(c, http.StatusInternalServerError, "api_error", "User context not found")
		return
	}

	reqLog := requestLogger(
		c,
		"handler.openai_gateway.videos",
		zap.Int64("user_id", subject.UserID),
		zap.Int64("api_key_id", apiKey.ID),
		zap.Any("group_id", apiKey.GroupID),
	)
	if !h.ensureResponsesDependencies(c, reqLog) {
		return
	}

	var body []byte
	var err error
	if c.Request.Method != http.MethodGet {
		body, err = pkghttputil.ReadRequestBodyWithPrealloc(c.Request)
		if err != nil {
			if maxErr, ok := extractMaxBytesError(err); ok {
				h.errorResponse(c, http.StatusRequestEntityTooLarge, "invalid_request_error", buildBodyTooLargeMessage(maxErr.Limit))
				return
			}
			h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
			return
		}
		if len(body) == 0 {
			h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Request body is empty")
			return
		}
	}

	parsed, err := h.gatewayService.ParseOpenAIVideosRequest(c, body)
	if err != nil {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	requestModel := parsed.Model
	videoPlatform := service.VideoGenerationPlatformForModel(requestModel)
	if apiKey.Group != nil && apiKey.Group.Platform != "" {
		videoPlatform = apiKey.Group.Platform
	}

	reqLog = reqLog.With(
		zap.String("model", requestModel),
		zap.String("request_id", parsed.RequestID),
		zap.String("platform", videoPlatform),
	)
	setOpsRequestContext(c, requestModel, false)
	setOpsEndpointContext(c, "", int16(service.RequestTypeFromLegacy(false, false)))

	var channelMapping service.ChannelMappingResult
	if !parsed.IsResult() {
		channelMapping, _ = h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, requestModel)
	}

	if h.errorPassthroughService != nil {
		service.BindErrorPassthroughService(c, h.errorPassthroughService)
	}

	subscription, _ := middleware2.GetSubscriptionFromContext(c)
	service.SetOpsLatencyMs(c, service.OpsAuthLatencyMsKey, time.Since(requestStart).Milliseconds())
	routingStart := time.Now()

	userReleaseFunc, acquired := h.acquireResponsesUserSlot(c, subject.UserID, subject.Concurrency, false, &streamStarted, reqLog)
	if !acquired {
		return
	}
	if userReleaseFunc != nil {
		defer userReleaseFunc()
	}

	if err := h.billingCacheService.CheckBillingEligibility(c.Request.Context(), apiKey.User, apiKey, apiKey.Group, subscription, service.QuotaPlatform(c.Request.Context(), apiKey)); err != nil {
		reqLog.Info("openai.videos.billing_eligibility_check_failed", zap.Error(err))
		status, code, message, retryAfter := billingErrorDetails(err)
		if retryAfter > 0 {
			c.Header("Retry-After", strconv.Itoa(retryAfter))
		}
		h.handleStreamingAwareError(c, status, code, message, streamStarted)
		return
	}

	sessionHash := h.gatewayService.GenerateExplicitSessionHash(c, body)
	if parsed.IsResult() {
		sessionHash = parsed.RequestID
	}
	requestCtx := c.Request.Context()

	maxAccountSwitches := h.maxAccountSwitches
	switchCount := 0
	failedAccountIDs := make(map[int64]struct{})
	sameAccountRetryCount := make(map[int64]int)
	var lastFailoverErr *service.UpstreamFailoverError

	for {
		reqLog.Debug("openai.videos.account_selecting", zap.Int("excluded_account_count", len(failedAccountIDs)))
		selection, scheduleDecision, err := h.gatewayService.SelectAccountWithSchedulerForVideos(
			requestCtx,
			apiKey.GroupID,
			sessionHash,
			requestModel,
			failedAccountIDs,
			videoPlatform,
		)
		if err != nil {
			reqLog.Warn("openai.videos.account_select_failed",
				zap.Error(err),
				zap.Int("excluded_account_count", len(failedAccountIDs)),
			)
			if len(failedAccountIDs) == 0 {
				cls := classifyNoAccountErrorFromGin(c, h.gatewayService, apiKey, requestModel, requestModel, videoPlatform)
				if !cls.ModelNotFound {
					markOpsRoutingCapacityLimitedIfNoAvailable(c, err)
				}
				message := cls.Message
				if !cls.ModelNotFound {
					message = "No available compatible accounts"
				}
				h.handleStreamingAwareError(c, cls.Status, cls.ErrType, message, streamStarted)
				return
			}
			if lastFailoverErr != nil {
				h.handleFailoverExhausted(c, lastFailoverErr, streamStarted)
			} else {
				h.handleFailoverExhaustedSimple(c, http.StatusBadGateway, streamStarted)
			}
			return
		}
		if selection == nil || selection.Account == nil {
			cls := classifyNoAccountErrorFromGin(c, h.gatewayService, apiKey, requestModel, requestModel, videoPlatform)
			if !cls.ModelNotFound {
				markOpsRoutingCapacityLimited(c)
			}
			message := cls.Message
			if !cls.ModelNotFound {
				message = "No available compatible accounts"
			}
			h.handleStreamingAwareError(c, cls.Status, cls.ErrType, message, streamStarted)
			return
		}

		reqLog.Debug("openai.videos.account_schedule_decision",
			zap.String("layer", scheduleDecision.Layer),
			zap.Bool("sticky_session_hit", scheduleDecision.StickySessionHit),
			zap.Int("candidate_count", scheduleDecision.CandidateCount),
			zap.Int("top_k", scheduleDecision.TopK),
			zap.Int64("latency_ms", scheduleDecision.LatencyMs),
			zap.Float64("load_skew", scheduleDecision.LoadSkew),
		)

		account := selection.Account
		sessionHash = ensureOpenAIPoolModeSessionHash(sessionHash, account)
		reqLog.Debug("openai.videos.account_selected", zap.Int64("account_id", account.ID), zap.String("account_name", account.Name))
		setOpsSelectedAccount(c, account.ID, account.Platform)

		accountReleaseFunc, acquired := h.acquireResponsesAccountSlot(c, apiKey.GroupID, sessionHash, selection, false, &streamStarted, reqLog)
		if !acquired {
			return
		}

		service.SetOpsLatencyMs(c, service.OpsRoutingLatencyMsKey, time.Since(routingStart).Milliseconds())
		forwardStart := time.Now()
		writerSizeBeforeForward := c.Writer.Size()
		result, err := func() (*service.OpenAIForwardResult, error) {
			defer func() {
				if accountReleaseFunc != nil {
					accountReleaseFunc()
				}
			}()
			return h.gatewayService.ForwardVideos(requestCtx, c, account, body, parsed, channelMapping.MappedModel)
		}()
		forwardDurationMs := time.Since(forwardStart).Milliseconds()
		upstreamLatencyMs, _ := getContextInt64(c, service.OpsUpstreamLatencyMsKey)
		responseLatencyMs := forwardDurationMs
		if upstreamLatencyMs > 0 && forwardDurationMs > upstreamLatencyMs {
			responseLatencyMs = forwardDurationMs - upstreamLatencyMs
		}
		service.SetOpsLatencyMs(c, service.OpsResponseLatencyMsKey, responseLatencyMs)

		if err != nil {
			var failoverErr *service.UpstreamFailoverError
			if errors.As(err, &failoverErr) {
				h.gatewayService.ReportOpenAIAccountScheduleResult(account.ID, false, nil)
				if c.Writer.Size() != writerSizeBeforeForward {
					reqLog.Warn("openai.videos.upstream_failover_skipped_after_flush",
						zap.Int64("account_id", account.ID),
						zap.Int("upstream_status", failoverErr.StatusCode),
					)
					h.handleFailoverExhausted(c, failoverErr, true)
					return
				}
				if failoverErr.RetryableOnSameAccount {
					retryLimit := account.GetPoolModeRetryCount()
					if sameAccountRetryCount[account.ID] < retryLimit {
						sameAccountRetryCount[account.ID]++
						reqLog.Warn("openai.videos.pool_mode_same_account_retry",
							zap.Int64("account_id", account.ID),
							zap.Int("upstream_status", failoverErr.StatusCode),
							zap.Int("retry_limit", retryLimit),
							zap.Int("retry_count", sameAccountRetryCount[account.ID]),
						)
						select {
						case <-requestCtx.Done():
							return
						case <-time.After(sameAccountRetryDelay):
						}
						continue
					}
				}
				h.gatewayService.RecordOpenAIAccountSwitch()
				failedAccountIDs[account.ID] = struct{}{}
				lastFailoverErr = failoverErr
				if switchCount >= maxAccountSwitches {
					h.handleFailoverExhausted(c, failoverErr, streamStarted)
					return
				}
				switchCount++
				if h.gatewayService.ShouldStopOpenAIOAuth429Failover(account, failoverErr.StatusCode, switchCount) {
					h.handleFailoverExhausted(c, failoverErr, streamStarted)
					return
				}
				reqLog.Warn("openai.videos.upstream_failover_switching",
					zap.Int64("account_id", account.ID),
					zap.Int("upstream_status", failoverErr.StatusCode),
					zap.Int("switch_count", switchCount),
					zap.Int("max_switches", maxAccountSwitches),
				)
				continue
			}
			h.gatewayService.ReportOpenAIAccountScheduleResult(account.ID, false, nil)
			upstreamErrorAlreadyCommunicated := openAIForwardErrorAlreadyCommunicated(c, writerSizeBeforeForward, err)
			wroteFallback := false
			if !upstreamErrorAlreadyCommunicated {
				wroteFallback = h.ensureForwardErrorResponse(c, streamStarted)
			}
			fields := []zap.Field{
				zap.Int64("account_id", account.ID),
				zap.Bool("fallback_error_response_written", wroteFallback),
				zap.Bool("upstream_error_response_already_written", upstreamErrorAlreadyCommunicated),
				zap.Error(err),
			}
			if shouldLogOpenAIForwardFailureAsWarn(c, wroteFallback) {
				reqLog.Warn("openai.videos.forward_failed", fields...)
				return
			}
			reqLog.Error("openai.videos.forward_failed", fields...)
			return
		}

		if result != nil {
			if account.Type == service.AccountTypeOAuth {
				h.gatewayService.UpdateCodexUsageSnapshotFromHeaders(c.Request.Context(), account.ID, result.ResponseHeaders)
			}
			h.gatewayService.ReportOpenAIAccountScheduleResult(account.ID, true, result.FirstTokenMs)
		} else {
			h.gatewayService.ReportOpenAIAccountScheduleResult(account.ID, true, nil)
		}

		userAgent := c.GetHeader("User-Agent")
		clientIP := ip.GetClientIP(c)
		requestPayloadHash := service.HashUsageRequestPayload(body)
		if parsed.IsResult() {
			requestPayloadHash = service.HashUsageRequestPayload([]byte(parsed.RequestID))
		}
		inboundEndpoint := GetInboundEndpoint(c)
		upstreamEndpoint := GetUpstreamEndpoint(c, account.Platform)

		upstreamModel := ""
		if result != nil {
			upstreamModel = result.UpstreamModel
		}
		h.submitMandatoryUsageRecordTask(c.Request.Context(), func(ctx context.Context) {
			if err := h.gatewayService.RecordUsage(ctx, &service.OpenAIRecordUsageInput{
				Result:             result,
				APIKey:             apiKey,
				User:               apiKey.User,
				Account:            account,
				Subscription:       subscription,
				InboundEndpoint:    inboundEndpoint,
				UpstreamEndpoint:   upstreamEndpoint,
				UserAgent:          userAgent,
				IPAddress:          clientIP,
				RequestPayloadHash: requestPayloadHash,
				APIKeyService:      h.apiKeyService,
				ChannelUsageFields: channelMapping.ToUsageFields(requestModel, upstreamModel),
			}); err != nil {
				logger.L().With(
					zap.String("component", "handler.openai_gateway.videos"),
					zap.Int64("user_id", subject.UserID),
					zap.Int64("api_key_id", apiKey.ID),
					zap.Any("group_id", apiKey.GroupID),
					zap.String("model", requestModel),
					zap.Int64("account_id", account.ID),
				).Error("openai.videos.record_usage_failed", zap.Error(err))
			}
		})

		reqLog.Debug("openai.videos.request_completed",
			zap.Int64("account_id", account.ID),
			zap.Int("switch_count", switchCount),
		)
		return
	}
}

export const BILLING_MODE_TOKEN = 'token'
export const BILLING_MODE_PER_REQUEST = 'per_request'
export const BILLING_MODE_IMAGE = 'image'
export const BILLING_MODE_VIDEO = 'video'

export function getBillingModeLabel(mode: string | null | undefined, t: (key: string) => string): string {
  switch (mode) {
    case BILLING_MODE_PER_REQUEST: return t('admin.usage.billingModePerRequest')
    case BILLING_MODE_IMAGE: return t('admin.usage.billingModeImage')
    case BILLING_MODE_VIDEO: return t('admin.usage.billingModeVideo')
    default: return t('admin.usage.billingModeToken')
  }
}

export function getBillingModeBadgeClass(mode: string | null | undefined): string {
  switch (mode) {
    case BILLING_MODE_PER_REQUEST: return 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300'
    case BILLING_MODE_IMAGE: return 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-300'
    case BILLING_MODE_VIDEO: return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'
    default: return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
  }
}

interface MediaBillingRow {
  image_count: number
  billing_mode?: string | null
  total_cost: number
  video_duration_seconds?: number | null
  video_unit_price?: number | null
  video_cost?: number | null
}

export function isImageUsage(row: Pick<MediaBillingRow, 'image_count' | 'billing_mode'> | null | undefined): boolean {
  return (row?.image_count ?? 0) > 0 && row?.billing_mode !== BILLING_MODE_TOKEN && row?.billing_mode !== BILLING_MODE_VIDEO
}

export function isVideoUsage(row: Pick<MediaBillingRow, 'billing_mode' | 'video_duration_seconds' | 'video_cost'> | null | undefined): boolean {
  return row?.billing_mode === BILLING_MODE_VIDEO || ((row?.video_duration_seconds ?? 0) > 0 || (row?.video_cost ?? 0) > 0) && row?.billing_mode !== BILLING_MODE_TOKEN
}

export function getDisplayBillingMode(row: Pick<MediaBillingRow, 'billing_mode' | 'image_count' | 'video_duration_seconds' | 'video_cost'> | null | undefined): string | null | undefined {
  if (isVideoUsage(row)) {
    return BILLING_MODE_VIDEO
  }
  if ((row?.image_count ?? 0) > 0 && !row?.billing_mode) {
    return BILLING_MODE_IMAGE
  }
  return row?.billing_mode
}

export function imageUnitPrice(row: Pick<MediaBillingRow, 'image_count' | 'total_cost'> | null): number {
  if (!row || row.image_count <= 0) return 0
  const total = row.total_cost ?? 0
  const price = total / row.image_count
  return Number.isFinite(price) ? price : 0
}

export function videoUnitPrice(row: Pick<MediaBillingRow, 'video_duration_seconds' | 'video_unit_price' | 'video_cost' | 'total_cost'> | null): number {
  if (!row) return 0
  if (row.video_unit_price != null) return row.video_unit_price
  const duration = row.video_duration_seconds ?? 0
  if (duration <= 0) return 0
  const total = row.video_cost ?? row.total_cost ?? 0
  const price = total / duration
  return Number.isFinite(price) ? price : 0
}

export function videoTotalCost(row: Pick<MediaBillingRow, 'video_cost' | 'total_cost'> | null): number {
  if (!row) return 0
  return row.video_cost ?? row.total_cost ?? 0
}

export function formatVideoDuration(seconds: number | null | undefined): string {
  const totalSeconds = Math.max(0, Math.round(seconds ?? 0))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const remainingSeconds = totalSeconds % 60

  const parts: string[] = []
  if (hours > 0) parts.push(`${hours}h`)
  if (minutes > 0) parts.push(`${minutes}m`)
  if (remainingSeconds > 0 || parts.length === 0) parts.push(`${remainingSeconds}s`)

  return parts.join(' ')
}

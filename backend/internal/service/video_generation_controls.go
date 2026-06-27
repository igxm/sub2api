package service

const videoGenerationPermissionMessage = "Video generation is not enabled for this group"

func GroupAllowsVideoGeneration(group *Group) bool {
	return group == nil || group.AllowVideoGeneration
}

func VideoGenerationPermissionMessage() string {
	return videoGenerationPermissionMessage
}

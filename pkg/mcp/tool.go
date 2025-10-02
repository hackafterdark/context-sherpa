package mcp

type ToolRequest struct {
	Params map[string]interface{}
}

type ToolResponse struct {
	Content []ToolContent
}

type ToolContent struct {
	Text string
}

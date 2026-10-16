// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package genai

// These tests exercise the REST transport end to end against a local
// httptest server, so they run without network access or an API key.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
)

const fakeAPIKey = "test-api-key"

// recordedRequest is an HTTP request received by the fake server.
type recordedRequest struct {
	method      string
	path        string
	contentType string
	apiKey      string
	body        map[string]any
}

// fakeServer is a local stand-in for the Generative Language REST API.
// It replies to every request with the same body and records each request.
type fakeServer struct {
	t        *testing.T
	response string

	mu       sync.Mutex
	requests []recordedRequest
}

func (s *fakeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	data, err := io.ReadAll(r.Body)
	if err != nil {
		s.t.Errorf("reading request body: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		s.t.Errorf("request body is not a JSON object: %v: %s", err, data)
	}
	apiKey := r.Header.Get("X-Goog-Api-Key")
	if apiKey == "" {
		apiKey = r.URL.Query().Get("key")
	}
	s.mu.Lock()
	s.requests = append(s.requests, recordedRequest{
		method:      r.Method,
		path:        r.URL.Path,
		contentType: r.Header.Get("Content-Type"),
		apiKey:      apiKey,
		body:        body,
	})
	s.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprint(w, s.response)
}

func (s *fakeServer) recorded() []recordedRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]recordedRequest(nil), s.requests...)
}

// newFakeClient starts a fake server that answers every request with
// response, and returns a Client that sends its REST requests there.
func newFakeClient(t *testing.T, response string) (*Client, *fakeServer) {
	t.Helper()
	fs := &fakeServer{t: t, response: response}
	srv := httptest.NewServer(fs)
	t.Cleanup(srv.Close)
	client, err := NewClient(context.Background(),
		option.WithEndpoint(srv.URL),
		option.WithAPIKey(fakeAPIKey))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { client.Close() })
	return client, fs
}

// toJSONValue converts v to the generic form produced by json.Unmarshal,
// so it can be compared with a decoded request body.
func toJSONValue(t *testing.T, v any) any {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestGenerateContentSendsAudioAsJSON(t *testing.T) {
	ctx := context.Background()
	client, fs := newFakeClient(t, `{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [{"text": "Hello Gemini"}]},
			"finishReason": "STOP"
		}],
		"usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
	}`)

	model := client.GenerativeModel("gemini-1.5-flash")
	model.SetTemperature(0)
	model.SetMaxOutputTokens(64)
	model.ResponseMIMEType = "text/plain"
	model.SystemInstruction = NewUserContent(Text("Transcribe the audio."))
	model.SafetySettings = []*SafetySetting{{
		Category:  HarmCategoryHarassment,
		Threshold: HarmBlockOnlyHigh,
	}}

	audio := []byte("RIFF")
	resp, err := model.GenerateContent(ctx, Blob{MIMEType: "audio/wav", Data: audio}, Text("Transcribe this."))
	if err != nil {
		t.Fatal(err)
	}

	reqs := fs.recorded()
	if len(reqs) != 1 {
		t.Fatalf("got %d requests, want 1", len(reqs))
	}
	req := reqs[0]
	if req.method != http.MethodPost {
		t.Errorf("method: got %q, want %q", req.method, http.MethodPost)
	}
	if want := "/v1beta/models/gemini-1.5-flash:generateContent"; req.path != want {
		t.Errorf("path: got %q, want %q", req.path, want)
	}
	if req.contentType != "application/json" {
		t.Errorf("content type: got %q, want %q", req.contentType, "application/json")
	}
	if req.apiKey != fakeAPIKey {
		t.Errorf("API key: got %q, want %q", req.apiKey, fakeAPIKey)
	}

	// The REST client encodes enums as numbers, bytes as base64 and
	// int32 values as JSON numbers.
	wantBody := toJSONValue(t, map[string]any{
		"model": "models/gemini-1.5-flash",
		"contents": []any{map[string]any{
			"role": "user",
			"parts": []any{
				map[string]any{"inlineData": map[string]any{"mimeType": "audio/wav", "data": audio}},
				map[string]any{"text": "Transcribe this."},
			},
		}},
		"safetySettings": []any{map[string]any{
			"category":  int(HarmCategoryHarassment),
			"threshold": int(HarmBlockOnlyHigh),
		}},
		"systemInstruction": map[string]any{
			"role":  "user",
			"parts": []any{map[string]any{"text": "Transcribe the audio."}},
		},
		"generationConfig": map[string]any{
			"temperature":      0,
			"maxOutputTokens":  64,
			"responseMimeType": "text/plain",
		},
	})
	if diff := cmp.Diff(wantBody, any(req.body)); diff != "" {
		t.Errorf("request body mismatch (-want +got):\n%s", diff)
	}

	want := &GenerateContentResponse{
		Candidates: []*Candidate{{
			Index:        0,
			Content:      &Content{Role: "model", Parts: []Part{Text("Hello Gemini")}},
			FinishReason: FinishReasonStop,
		}},
		UsageMetadata: &UsageMetadata{PromptTokenCount: 3, CandidatesTokenCount: 2, TotalTokenCount: 5},
	}
	if diff := cmp.Diff(want, resp); diff != "" {
		t.Errorf("response mismatch (-want +got):\n%s", diff)
	}
}

func TestGenerateContentDecodesNullFields(t *testing.T) {
	ctx := context.Background()
	client, _ := newFakeClient(t, `{
		"candidates": [{
			"index": 0,
			"content": {"role": "model", "parts": [{"text": "ok"}]},
			"finishReason": "STOP",
			"safetyRatings": null,
			"citationMetadata": null,
			"groundingAttributions": null
		}],
		"promptFeedback": null,
		"usageMetadata": null
	}`)

	resp, err := client.GenerativeModel("gemini-1.5-flash").GenerateContent(ctx, Text("hi"))
	if err != nil {
		t.Fatal(err)
	}
	want := &GenerateContentResponse{
		Candidates: []*Candidate{{
			Index:        0,
			Content:      &Content{Role: "model", Parts: []Part{Text("ok")}},
			FinishReason: FinishReasonStop,
		}},
	}
	if diff := cmp.Diff(want, resp); diff != "" {
		t.Errorf("response mismatch (-want +got):\n%s", diff)
	}
}

func TestGenerateContentStreamPreservesUsageMetadata(t *testing.T) {
	ctx := context.Background()
	client, fs := newFakeClient(t, `[
		{"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "Hel"}]}}]},
		{
			"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "lo"}]}, "finishReason": "STOP"}],
			"usageMetadata": {
				"promptTokenCount": 7,
				"cachedContentTokenCount": 4,
				"candidatesTokenCount": 11,
				"totalTokenCount": 18
			}
		}
	]`)

	iter := client.GenerativeModel("gemini-1.5-flash").GenerateContentStream(ctx, Text("hi"))
	resps, err := all(iter)
	if err != nil {
		t.Fatal(err)
	}
	if len(resps) != 2 {
		t.Fatalf("got %d streamed responses, want 2", len(resps))
	}
	if resps[0].UsageMetadata != nil {
		t.Errorf("first chunk: got usage %+v, want nil", resps[0].UsageMetadata)
	}
	wantUsage := &UsageMetadata{
		PromptTokenCount:        7,
		CachedContentTokenCount: 4,
		CandidatesTokenCount:    11,
		TotalTokenCount:         18,
	}
	if diff := cmp.Diff(wantUsage, resps[1].UsageMetadata); diff != "" {
		t.Errorf("last chunk usage mismatch (-want +got):\n%s", diff)
	}

	reqs := fs.recorded()
	if len(reqs) != 1 {
		t.Fatalf("got %d requests, want 1", len(reqs))
	}
	if want := "/v1beta/models/gemini-1.5-flash:streamGenerateContent"; reqs[0].path != want {
		t.Errorf("path: got %q, want %q", reqs[0].path, want)
	}
}

func TestFunctionCallsAndFinishReason(t *testing.T) {
	probeTool := &Tool{FunctionDeclarations: []*FunctionDeclaration{{
		Name:        "ProbeTool",
		Description: "Probes a value.",
		Parameters: &Schema{
			Type:       TypeObject,
			Properties: map[string]*Schema{"value": {Type: TypeNumber}},
			Required:   []string{"value"},
		},
	}}}
	textPart := `{"text": "Calling the probe tool."}`
	callPart := `{"functionCall": {"name": "ProbeTool", "args": {"value": 1}}}`

	for _, test := range []struct {
		stream   bool
		text     bool
		call     bool
		wantText string
		wantCall bool
	}{
		{stream: true, text: true, call: true, wantText: "Calling the probe tool.", wantCall: true},
		{stream: true, text: false, call: true, wantCall: true},
		{stream: true, text: true, call: false, wantText: "Calling the probe tool."},
		{stream: false, text: true, call: true, wantText: "Calling the probe tool.", wantCall: true},
	} {
		mode := "generate"
		if test.stream {
			mode = "stream"
		}
		name := fmt.Sprintf("%s/text=%t,call=%t", mode, test.text, test.call)
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()

			// The server sends the text and the function call in separate
			// chunks when streaming, and in one candidate otherwise.
			var parts []string
			if test.text {
				parts = append(parts, textPart)
			}
			if test.call {
				parts = append(parts, callPart)
			}
			candidate := func(parts []string, finish string) string {
				c := `{"index": 0, "content": {"role": "model", "parts": [`
				for i, p := range parts {
					if i > 0 {
						c += ","
					}
					c += p
				}
				c += `]}`
				if finish != "" {
					c += `, "finishReason": "` + finish + `"`
				}
				return c + `}`
			}
			var body string
			if test.stream {
				body = "["
				for i, p := range parts {
					finish := ""
					if i == len(parts)-1 {
						finish = "STOP"
					}
					if i > 0 {
						body += ","
					}
					body += `{"candidates": [` + candidate([]string{p}, finish) + `]}`
				}
				body += "]"
			} else {
				body = `{"candidates": [` + candidate(parts, "STOP") + `]}`
			}

			client, fs := newFakeClient(t, body)
			model := client.GenerativeModel("gemini-1.5-pro")
			model.Tools = []*Tool{probeTool}

			var resp *GenerateContentResponse
			if test.stream {
				iter := model.GenerateContentStream(ctx, Text("Use the probe tool"))
				if _, err := all(iter); err != nil {
					t.Fatal(err)
				}
				resp = iter.MergedResponse()
			} else {
				var err error
				resp, err = model.GenerateContent(ctx, Text("Use the probe tool"))
				if err != nil {
					t.Fatal(err)
				}
			}

			if len(resp.Candidates) != 1 {
				t.Fatalf("got %d candidates, want 1", len(resp.Candidates))
			}
			c := resp.Candidates[0]
			var wantCalls []FunctionCall
			if test.wantCall {
				wantCalls = []FunctionCall{{Name: "ProbeTool", Args: map[string]any{"value": float64(1)}}}
			}
			if diff := cmp.Diff(wantCalls, c.FunctionCalls()); diff != "" {
				t.Errorf("function calls mismatch (-want +got):\n%s", diff)
			}
			var gotText string
			for _, p := range c.Content.Parts {
				if s, ok := p.(Text); ok {
					gotText += string(s)
				}
			}
			if gotText != test.wantText {
				t.Errorf("text: got %q, want %q", gotText, test.wantText)
			}
			// The API reports STOP even when the model calls a function;
			// the client must pass it through unchanged.
			if c.FinishReason != FinishReasonStop {
				t.Errorf("finish reason: got %v, want %v", c.FinishReason, FinishReasonStop)
			}

			reqs := fs.recorded()
			if len(reqs) != 1 {
				t.Fatalf("got %d requests, want 1", len(reqs))
			}
			wantTools := toJSONValue(t, []any{map[string]any{
				"functionDeclarations": []any{map[string]any{
					"name":        "ProbeTool",
					"description": "Probes a value.",
					"parameters": map[string]any{
						"type":       int(TypeObject),
						"properties": map[string]any{"value": map[string]any{"type": int(TypeNumber)}},
						"required":   []any{"value"},
					},
				}},
			}})
			if diff := cmp.Diff(wantTools, reqs[0].body["tools"]); diff != "" {
				t.Errorf("request tools mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

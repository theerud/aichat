use super::*;

use anyhow::{bail, Context, Result};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

const API_BASE: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct OpenAIResponsesConfig {
    pub name: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub organization_id: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl OpenAIResponsesClient {
    config_get_fn!(api_key, get_api_key);
    config_get_fn!(api_base, get_api_base);

    pub const PROMPTS: [PromptAction<'static>; 1] = [("api_key", "API Key", None)];
}

impl_client_trait!(
    OpenAIResponsesClient,
    (
        prepare_chat_completions,
        openai_responses,
        openai_responses_streaming
    ),
    (prepare_embeddings, openai_embeddings),
    (noop_prepare_rerank, noop_rerank),
);

fn prepare_chat_completions(
    self_: &OpenAIResponsesClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());

    let url = format!("{}/responses", api_base.trim_end_matches('/'));

    let body = openai_build_responses_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

fn prepare_embeddings(self_: &OpenAIResponsesClient, data: &EmbeddingsData) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());

    let url = format!("{api_base}/embeddings");

    let body = openai_build_embeddings_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

pub async fn openai_responses(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<ChatCompletionsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }

    debug!("non-stream-data: {data}");
    openai_extract_responses(&data)
}

pub async fn openai_responses_streaming(
    builder: RequestBuilder,
    handler: &mut SseHandler,
    _model: &Model,
) -> Result<()> {
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut current_tool_call_id = String::new();
    let mut current_tool_function_name = String::new();
    let mut current_tool_function_arguments = String::new();

    let handle = |message: SseMmessage| -> Result<bool> {
        if message.data == "[DONE]" {
            if !current_tool_function_name.is_empty() {
                let arguments: Value = serde_json::from_str(&current_tool_function_arguments).unwrap_or_default();
                tool_calls.push(ToolCall::new(
                    current_tool_function_name.clone(),
                    arguments,
                    Some(current_tool_call_id.clone()),
                ));
            }
            return Ok(true);
        }

        let data: Value = serde_json::from_str(&message.data)?;
        debug!("stream-data: {data}");

        match data["type"].as_str() {
            Some("response.output_text.delta") => {
                if let Some(text) = data["delta"].as_str() {
                    handler.text(text)?;
                }
            }
            Some("response.tool_code.delta") => {
                if let Some(id) = data["item_id"].as_str() {
                    if current_tool_call_id != id {
                        if !current_tool_function_name.is_empty() {
                            let arguments: Value = serde_json::from_str(&current_tool_function_arguments).unwrap_or_default();
                            tool_calls.push(ToolCall::new(
                                current_tool_function_name.clone(),
                                arguments,
                                Some(current_tool_call_id.clone()),
                            ));
                        }
                        current_tool_call_id = id.to_string();
                        current_tool_function_name.clear();
                        current_tool_function_arguments.clear();
                    }
                }
                if let Some(part) = data["part"].as_object() {
                    if let Some(name) = part["function_name"].as_str() {
                        current_tool_function_name.push_str(name);
                    }
                    if let Some(args_chunk) = part["args_chunk"].as_str() {
                        current_tool_function_arguments.push_str(args_chunk);
                    }
                }
            }
            _ => {}
        }
        Ok(false)
    };

    sse_stream(builder, handle).await?;

    if !tool_calls.is_empty() {
        for tool_call in tool_calls {
            handler.tool_call(tool_call)?;
        }
    }

    Ok(())
}

pub fn openai_build_responses_body(data: ChatCompletionsData, model: &Model) -> Value {
    let ChatCompletionsData {
        messages,
        temperature,
        top_p,
        functions,
        stream,
    } = data;

    let mut instructions = String::new();
    let mut input = String::new();
    let mut tool_outputs = Vec::new();
    let mut previous_response_id = None;

    for message in messages {
        match message.role {
            MessageRole::System => instructions.push_str(&message.content.to_text()),
            MessageRole::User => input.push_str(&message.content.to_text()),
            MessageRole::Assistant => {
                 if let MessageContent::Text(text) = message.content {
                    if text.starts_with("response_id:") {
                        previous_response_id = Some(text.strip_prefix("response_id:").unwrap().trim().to_string());
                    } else {
                        input.push_str(&text);
                    }
                }
            }
            MessageRole::Tool => {
                if let MessageContent::ToolCalls(tool_calls) = message.content {
                    for result in tool_calls.tool_results {
                        tool_outputs.push(json!({
                            "tool_call_id": result.call.id,
                            "output": result.output
                        }));
                    }
                }
            }
        }
    }

    let mut body = json!({
        "model": &model.real_name(),
        "input": input,
    });

    if !instructions.is_empty() {
        body["instructions"] = instructions.into();
    }
    if !tool_outputs.is_empty() {
        body["tool_outputs"] = json!(tool_outputs);
    }
    if let Some(id) = previous_response_id {
        body["previous_response_id"] = id.into();
    }
    if let Some(v) = temperature {
        body["temperature"] = v.into();
    }
    if let Some(v) = top_p {
        body["top_p"] = v.into();
    }
    if stream {
        body["stream"] = true.into();
    }
    if let Some(functions) = functions {
        body["tools"] = functions
            .into_iter()
            .map(|v| {
                json!({
                    "type": "function",
                    "function": v,
                })
            })
            .collect::<Value>();
    }

    body
}

pub fn openai_extract_responses(data: &Value) -> Result<ChatCompletionsOutput> {
    let mut text = String::new();
    let mut tool_calls = vec![];

    if let Some(response) = data.get("response") {
        if let Some(outputs) = response["output"].as_array() {
            for output in outputs {
                if let Some(content) = output["content"].as_array() {
                    for part in content {
                        if part["type"] == "output_text" {
                            if let Some(value) = part["text"].as_str() {
                                text.push_str(value);
                            }
                        } else if part["type"] == "tool_code" {
                            if let (Some(id), Some(function)) = (part["id"].as_str(), part["function"].as_object()) {
                                let name = function["name"].as_str().unwrap_or_default().to_string();
                                let arguments = function["args"].as_str().unwrap_or_default().to_string();
                                let arguments: Value = serde_json::from_str(&arguments).unwrap_or_default();
                                tool_calls.push(ToolCall::new(name, arguments, Some(id.to_string())));
                            }
                        }
                    }
                }
            }
        }
    }

    if text.is_empty() && tool_calls.is_empty() {
        bail!("Invalid response data: {data}");
    }

    let output = ChatCompletionsOutput {
        text,
        tool_calls,
        id: data["id"].as_str().map(|v| v.to_string()),
        input_tokens: data["usage"]["input_tokens"].as_u64(),
        output_tokens: data["usage"]["output_tokens"].as_u64(),
    };
    Ok(output)
}


// The following are not used by the Responses API, but are kept for trait completeness
pub async fn openai_embeddings(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<EmbeddingsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    let res_body: EmbeddingsResBody =
        serde_json::from_value(data).context("Invalid embeddings data")?;
    let output = res_body.data.into_iter().map(|v| v.embedding).collect();
    Ok(output)
}

#[derive(Deserialize)]
struct EmbeddingsResBody {
    data: Vec<EmbeddingsResBodyEmbedding>,
}

#[derive(Deserialize)]
struct EmbeddingsResBodyEmbedding {
    embedding: Vec<f32>,
}

pub fn openai_build_embeddings_body(data: &EmbeddingsData, model: &Model) -> Value {
    json!({
        "input": data.texts,
        "model": model.real_name()
    })
}

use super::*;

use anyhow::{bail, Result};
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
    (noop_prepare_embeddings, noop_embeddings),
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
        functions: _,
        stream,
    } = data;

    let (_, history_messages) = messages.split_last().unzip();

    let (instructions, previous_response_id) = extract_history(history_messages.unwrap_or_default());

    let input = build_request_input(&messages);

    let mut body = json!({
        "model": &model.real_name(),
        "input": input,
    });

    if let Some(instructions) = instructions {
        body["instructions"] = instructions.into();
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

    body
}

fn extract_history(messages: &[Message]) -> (Option<String>, Option<String>) {
    let mut instructions = None;
    let mut previous_response_id = None;

    for message in messages {
        if message.role.is_system() {
            instructions = Some(message.content.to_text());
        } else if message.role.is_assistant() {
            if let MessageContent::Text(text) = &message.content {
                if let Some(id) = text.strip_prefix("id:").and_then(|s| s.split('\n').next()) {
                    previous_response_id = Some(id.trim().to_string());
                }
            }
        }
    }

    (instructions, previous_response_id)
}

fn build_request_input(messages: &Vec<Message>) -> Value {
    if messages.len() == 1 {
        if let MessageContent::Text(text) = &messages[0].content {
            return json!(text);
        }
    }
    json!(messages.iter().map(|message| match (&message.role, &message.content) {
        (role, MessageContent::Text(text)) => json!({
            "role": role, "content": text
        }),
        (role, MessageContent::Array(list)) => {
            let content: Vec<Value> = list
                .iter()
                .map(|item| {
                    let direction = match role {
                        MessageRole::Assistant => "output",
                        _ => "input",
                    };
                    match item {
                        MessageContentPart::Text { text } => json!({
                            "type": format!("{direction}_text"),
                            "text": text
                        }),
                        MessageContentPart::ImageUrl { image_url } => json!({
                            "type": format!("{direction}_image"),
                            "image_url": image_url.url,
                            "detail": "auto"
                        }),
                    }
                })
                .collect();
            json!({"role": role, "content": content})
        }
        (_, MessageContent::ToolCalls(tool_calls)) => {
            let tool_outputs: Vec<Value> = tool_calls
                .tool_results
                .iter()
                .map(|result| {
                    json!({
                        "tool_call_id": result.call.id,
                        "output": result.output
                    })
                })
                .collect();
            json!([{"role": "user", "content": [{"type": "tool_outputs", "tool_outputs": tool_outputs}]}])
        }
    }).collect::<Value>())
}


pub fn openai_extract_responses(data: &Value) -> Result<ChatCompletionsOutput> {
    let text = data["output"][0]["content"][0]["text"].as_str().unwrap_or_default().to_string();

    if text.is_empty() {
        bail!("Invalid response data: {data}");
    }

    let output = ChatCompletionsOutput {
        text,
        tool_calls: vec![],
        id: data.get("id").and_then(|v| v.as_str()).map(|v| v.to_string()),
        input_tokens: data
            .get("usage")
            .and_then(|u| u.get("input_tokens"))
            .and_then(|v| v.as_u64()),
        output_tokens: data
            .get("usage")
            .and_then(|u| u.get("output_tokens"))
            .and_then(|v| v.as_u64()),
    };
    Ok(output)
}




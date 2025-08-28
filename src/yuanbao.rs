use crate::service::Config;
use anyhow::{Context, Error, anyhow, bail};
use async_channel::{Receiver, Sender, unbounded};
use axum::http::HeaderValue;
use futures_util::StreamExt;
use reqwest::Client;
use reqwest::header::{HeaderMap, HeaderName};
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt::{Debug, Display, Formatter};
use std::str::FromStr;
use std::time::Duration;
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

static CREATE_URL: &str = "https://yuanbao.tencent.com/api/user/agent/conversation/create";
static CLEAR_URL: &str = "https://yuanbao.tencent.com/api/user/agent/conversation/v1/clear";
static CHAT_URL: &str = "https://yuanbao.tencent.com/api/chat/{}";

#[derive(Debug)]
pub enum ChatCompletionEvent {
    Message(ChatCompletionMessage),
    Error(Error),
    Finish(String),
}
#[derive(Debug)]
pub struct ChatCompletionMessage {
    pub r#type: ChatCompletionMessageType,
    pub text: String,
}
#[derive(Debug)]
pub enum ChatCompletionMessageType {
    Think,
    Msg,
}
pub struct ChatCompletionRequest {
    pub messages: ChatMessages,
    pub chat_model: ChatModel,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessages(pub Vec<ChatMessage>);
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
}
impl Display for ChatMessages {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let arr = &self.0;
        if arr.len() == 0 {
            return Err(std::fmt::Error);
        }
        if arr.len() == 1 {
            write!(f, "{}", arr[0].content.as_ref().unwrap_or(&"".to_string()))?;
            return Ok(());
        }
        for item in arr {
            write!(
                f,
                "#[{}]\n{}\n\n",
                item.role.trim(),
                item.content.as_ref().unwrap_or(&"".to_string()).trim()
            )?;
        }
        Ok(())
    }
}
#[derive(Copy, Clone)]
pub enum ChatModel {
    DeepSeekV3,
    DeepSeekR1,
    DeepSeekV3Search, // 新增搜索模型
    DeepSeekR1Search, // 新增搜索模型
}
impl FromStr for ChatModel {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "deepseek-r1" => Ok(ChatModel::DeepSeekR1),
            "deepseek-v3" => Ok(ChatModel::DeepSeekV3),
            "deepseek-r1-search" => Ok(ChatModel::DeepSeekR1Search), // 新增解析
            "deepseek-v3-search" => Ok(ChatModel::DeepSeekV3Search), // 新增解析
            &_ => {
                bail!("invalid model")
            }
        }
    }
}
impl ChatModel {
    pub fn to_yuanbao_string(&self) -> String {
        match self {
            ChatModel::DeepSeekV3 => "deep_seek_v3",
            ChatModel::DeepSeekR1 => "deep_seek",
            ChatModel::DeepSeekV3Search => "deep_seek_v3", // 搜索模型使用相同基础名称
            ChatModel::DeepSeekR1Search => "deep_seek",    // 搜索模型使用相同基础名称
        }
        .to_string()
    }
    pub fn to_common_string(&self) -> String {
        match self {
            ChatModel::DeepSeekV3 => "deepseek-v3",
            ChatModel::DeepSeekR1 => "deepseek-r1",
            ChatModel::DeepSeekV3Search => "deepseek-v3-search", // 新增
            ChatModel::DeepSeekR1Search => "deepseek-r1-search", // 新增
        }
        .to_string()
    }
    
    // 新增方法判断是否为搜索模型
    pub fn is_search_model(&self) -> bool {
        matches!(self, ChatModel::DeepSeekV3Search | ChatModel::DeepSeekR1Search)
    }
}

#[derive(Clone)]
pub struct Yuanbao {
    config: Config,
    client: Client,
}
impl Yuanbao {
    pub fn new(config: Config) -> Yuanbao {
        let headers = Self::make_headers(&config);
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .unwrap();
        Yuanbao { config, client }
    }
    pub async fn create_conversation(&self) -> anyhow::Result<String> {
        let res: serde_json::Value = self
            .client
            .post(CREATE_URL)
            .json(&json!({"agentId":self.config.agent_id}))
            .send()
            .await
            .context("cannot send request")?
            .error_for_status()
            .context("error status code")?
            .json()
            .await
            .context("cannot parse json")?;
        let id = res["id"].as_str().context("id not in json")?;
        Ok(id.to_string())
    }
    pub async fn create_completion(
        &self,
        request: ChatCompletionRequest,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<Receiver<ChatCompletionEvent>> {
        info!("Creating conversation");
        let conversation_id = self
            .create_conversation()
            .await
            .context("cannot create conversation")?;
        info!("Conversation id: {}", conversation_id);
        let prompt = request.messages.to_string();
        
        // 构建基础请求体
        let mut body = json!(
            {
        "model": "gpt_175B_0404",
        "prompt": prompt,
        "plugin": "Adaptive",
        "displayPrompt": prompt,
        "displayPromptType": 1,
        "options": {"imageIntention": {"needIntentionModel": true, "backendUpdateFlag": 2, "intentionStatus": true}},
        "multimedia": [],
        "agentId": self.config.agent_id,
        "supportHint": 1,
        "version": "v2",
        "chatModelId": request.chat_model.to_yuanbao_string(),
            }
        );
        
        // 如果是搜索模型，添加 supportFunctions 字段
        if request.chat_model.is_search_model() {
            body["supportFunctions"] = json!(["openInternetSearch"]);
        }
        
        let formatted_url = CHAT_URL.replace("{}", &conversation_id);
        let mut sse = EventSource::new(self.client.post(&formatted_url).json(&body))
            .context("failed to get next event")?;
        let (sender, receiver) = unbounded::<ChatCompletionEvent>();

        tokio::spawn(async move {
            Self::process_sse(&mut sse, sender, cancel_token).await;
        });
        Ok(receiver)
    }
    async fn process_sse(
        sse: &mut EventSource,
        sender: Sender<ChatCompletionEvent>,
        cancel_token: CancellationToken,
    ) {
        let mut finish_reason = "stop".to_string();
        loop {
            let event;
            select! {
                Some(e)=sse.next()=>{
                    event=e;
                },
                _=cancel_token.cancelled()=>{
                    info!("Stream cancelled");
                    break;
                },
                else => {
                    info!("Stream ended (pattern else)");
                    break;
                }
            }
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(message)) => {
                    if message.event != "message" {
                        continue;
                    }
                    let res = serde_json::from_str::<serde_json::Value>(&message.data);
                    let value;
                    match res {
                        Ok(v) => {
                            value = v;
                        }
                        Err(_) => continue,
                    }
                    match value["type"].as_str().unwrap_or("") {
                        "think" => {
                            let content = value["content"].as_str().unwrap_or("");
                            if content == "" {
                                continue;
                            }
                            let _ = sender
                                .send(ChatCompletionEvent::Message(ChatCompletionMessage {
                                    r#type: ChatCompletionMessageType::Think,
                                    text: content.to_string(),
                                }))
                                .await;
                        }
                        "text" => {
                            let msg = value["msg"].as_str().unwrap_or("");
                            let _ = sender
                                .send(ChatCompletionEvent::Message(ChatCompletionMessage {
                                    r#type: ChatCompletionMessageType::Msg,
                                    text: msg.to_string(),
                                }))
                                .await;
                        }
                        _ => {
                            let stop_reason = value["stopReason"].as_str().unwrap_or("");
                            if stop_reason != "" {
                                finish_reason = stop_reason.to_string();
                            }
                        }
                    }
                    debug!(?message, "Event message");
                }
                Err(err) => match err {
                    reqwest_eventsource::Error::StreamEnded => {
                        info!("Stream ended (error type)");
                        break;
                    }
                    _ => {
                        let _ = sender
                            .send(ChatCompletionEvent::Error(anyhow!(
                                "Error on stream: {}",
                                err
                            )))
                            .await;
                    }
                },
            }
        }
        let _ = sender
            .send(ChatCompletionEvent::Finish(finish_reason))
            .await;
    }
    fn make_headers(config: &Config) -> HeaderMap {
        HeaderMap::from_iter(
            vec![
                (
                    HeaderName::from_str("Cookie").unwrap(),
                    HeaderValue::from_str(&format!(
                        "hy_source=web; hy_user={}; hy_token={}",
                        config.hy_user, config.hy_token
                    ))
                    .unwrap(),
                ),
                (
                    HeaderName::from_str("Origin").unwrap(),
                    HeaderValue::from_str("https://yuanbao.tencent.com").unwrap(),
                ),
                (
                    HeaderName::from_str("Referer").unwrap(),
                    HeaderValue::from_str(&format!(
                        "https://yuanbao.tencent.com/chat/{}",
                        config.agent_id
                    ))
                    .unwrap(),
                ),
                (
                    HeaderName::from_str("X-Agentid").unwrap(),
                    HeaderValue::from_str(&config.agent_id).unwrap(),
                ),
                (
                    HeaderName::from_str("User-Agent").unwrap(),
                    HeaderValue::from_str(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
                     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
                    )
                    .unwrap(),
                ),
            ]
            .into_iter(),
        )
    }
}

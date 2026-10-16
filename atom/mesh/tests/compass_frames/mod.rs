//! The clock wire's length-prefixed JSON frames, for scripted authorities.

use serde_json::Value;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

pub async fn read_frame(sock: &mut TcpStream) -> Value {
    let mut body = vec![0; sock.read_u32().await.unwrap() as usize];
    sock.read_exact(&mut body).await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

pub async fn write_frame(sock: &mut TcpStream, message: Value) {
    let body = message.to_string();
    sock.write_u32(body.len() as u32).await.unwrap();
    sock.write_all(body.as_bytes()).await.unwrap();
}

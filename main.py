import os
import asyncio
import logging
import re
from typing import List, Optional, Set, Dict
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration 
class Config(BaseModel):
    group: str
    cookie: str
    webhook: Optional[HttpUrl] = None
    webhook_username: str = Field(default="Group Wall Filter")
    webhook_image: Optional[HttpUrl] = None
    openai_api_key: str
    filter_list: List[str] = [
        "This is an offical message from Roblox", "Free Robux",
    ]
    link_pattern: str = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»""'']))'
    whitelisted_users: Set[int] = Field(default_factory=set)
    cache_file: str = "last_processed_post.txt"
    batchlimit: int = 100
    cycle_time: int = 60

config = Config(
    group=os.getenv("GROUP", ""),
    cookie=os.getenv("COOKIE", ""),
    webhook=os.getenv("WEBHOOK"),
    webhook_username=os.getenv("WEBHOOK_USERNAME", "Group Wall Filter"),
    webhook_image=os.getenv("WEBHOOK_IMAGE"),
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    whitelisted_users=set(map(int, os.getenv("WHITELISTED_USERS", "").split(","))) if os.getenv("WHITELISTED_USERS") else set()
)

link_regex = re.compile(config.link_pattern, re.IGNORECASE)

last_processed_post_id: str = ""

def load_cache():
    global last_processed_post_id
    try:
        if os.path.exists(config.cache_file):
            with open(config.cache_file, 'r') as f:
                last_processed_post_id = f.read().strip()
            if last_processed_post_id:
                logger.info(f"Loaded last processed post ID: {last_processed_post_id}")
            else:
                logger.info("Cache file exists but is empty. Starting with empty cache.")
        else:
            logger.info("No cache file found. Starting with empty cache.")
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        last_processed_post_id = ""

async def save_cache():
    global last_processed_post_id
    if last_processed_post_id:
        try:
            with open(config.cache_file, 'w') as f:
                f.write(str(last_processed_post_id))
            logger.info(f"Saved last processed post ID: {last_processed_post_id}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    else:
        logger.warning("No post ID to save. Cache file not updated.")


async def get_token(session: aiohttp.ClientSession) -> Optional[str]:
    auth_url = "https://auth.roblox.com/v2/logout"
    try:
        async with session.post(auth_url) as response:
            return response.headers.get("x-csrf-token")
    except aiohttp.ClientError as e:
        logger.error(f"Failed to get token: {e}")
        return None

async def check_openai_moderation(session: aiohttp.ClientSession, text: str) -> bool:
    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.openai_api_key}"
    }
    data = {"input": text}

    try:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                flagged = result["results"][0]["flagged"]
                logger.info(f"OpenAI moderation result for '{text}': {'Flagged' if flagged else 'Not flagged'}")
                return flagged
            else:
                logger.error(f"OpenAI API request failed with status {response.status}")
                return False
    except aiohttp.ClientError as e:
        logger.error(f"Error occurred while checking OpenAI moderation: {e}")
        return False

def contains_link(text: str) -> bool:
    if link_regex.search(text):
        return True
    
    words = text.split()
    for word in words:
        if '.' in word and not word.startswith('.') and not word.endswith('.'):
            parts = word.split('.')
            if len(parts) == 2 and len(parts[1]) >= 2:
                return True
    
    return False

async def filter_wall(session: aiohttp.ClientSession, token: str):
    global last_processed_post_id
    headers = {
        "X-CSRF-TOKEN": token,
        "Content-Type": "application/json"
    }

    roblox_url = f"https://groups.roblox.com/v1/groups/{config.group}/wall/posts?limit={config.batchlimit}&sortOrder=Desc"

    try:
        async with session.get(roblox_url, headers=headers) as response:
            if response.status != 200:
                logger.error(f"Failed to get group wall posts: {response.status}")
                return

            posts = (await response.json())["data"]

            if not posts:
                logger.info("No posts found.")
                return

            logger.info(f"Fetched {len(posts)} posts. Last processed post ID: {last_processed_post_id}")

            new_posts = []
            for post in posts:
                if str(post["id"]) == last_processed_post_id:
                    logger.info(f"Reached last processed post: {last_processed_post_id}")
                    break
                new_posts.append(post)

            if not new_posts:
                logger.info("No new posts to process.")
                return

            logger.info(f"Processing {len(new_posts)} new posts")

            for post in reversed(new_posts):
                post_id = post["id"]
                body_text = post["body"]
                user_id = post["poster"]["userId"]

                logger.info(f"Processing post {post_id}: {body_text[:50]}...")

                if user_id in config.whitelisted_users:
                    logger.info(f"User {user_id} is whitelisted. Skipping filters.")
                    continue

                detected_word = next((word for word in config.filter_list if word.lower() in body_text.lower()), None)
                link_detected = contains_link(body_text)

                if detected_word:
                    logger.info(f"Filtered word detected: {detected_word}")
                    await delete_post(session, headers, post, f"Filtered word: {detected_word}")
                elif link_detected:
                    logger.info("Link detected in post")
                    await delete_post(session, headers, post, "Link detected")
                elif await check_openai_moderation(session, body_text):
                    logger.info("OpenAI moderation flagged the post")
                    await delete_post(session, headers, post, "OpenAI moderation")
                else:
                    logger.info("Post passed all checks")

            if new_posts:
                last_processed_post_id = new_posts[0]["id"]
                logger.info(f"Updating last processed post ID to: {last_processed_post_id}")
                await save_cache()
            else:
                logger.info("No new posts processed. Cache not updated.")

    except aiohttp.ClientError as e:
        logger.error(f"Error occurred while filtering wall: {e}")

async def delete_post(session: aiohttp.ClientSession, headers: dict, post: dict, detected_word: str):
    post_id = post["id"]
    username = post["poster"]["username"]
    user_id = post["poster"]["userId"]
    body_text = post["body"]

    delete_url = f"https://groups.roblox.com/v1/groups/{config.group}/wall/posts/{post_id}"

    try:
        async with session.delete(delete_url, headers=headers) as response:
            if response.status == 200:
                logger.info(f"Deleted filtered message from {username}")
                await send_discord_webhook(username, user_id, body_text, detected_word)
            else:
                logger.error(f"Failed to delete post: {response.status}")
    except aiohttp.ClientError as e:
        logger.error(f"Error occurred while deleting post: {e}")

async def send_discord_webhook(username: str, user_id: int, body_text: str, detected_word: str):
    if not config.webhook:
        logger.info(f"Webhook not configured. Skipping notification for filtered message from {username}")
        return

    discord_data = {
        "username": config.webhook_username,
        "avatar_url": str(config.webhook_image) if config.webhook_image else None,
        "content": "**Message Filtered from Group Wall!**",
        "embeds": [{
            "type": "rich",
            "title": f"{username}'s Message was Removed!",
            "description": f"**Message Body**\n```{body_text}```\n**Filtered String:** `{detected_word}`\n\n**Profile Link**\nhttps://www.roblox.com/users/{user_id}/profile\n",
            "color": 16711902
        }]
    }

    async with aiohttp.ClientSession() as session:
        try:
            webhook_url = str(config.webhook)
            async with session.post(webhook_url, json=discord_data) as response:
                if response.status != 204:
                    logger.error(f"Failed to send Discord webhook: {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Error occurred while sending Discord webhook: {e}")

async def run_filter():
    async with aiohttp.ClientSession(cookies={'.ROBLOSECURITY': config.cookie}) as session:
        token = await get_token(session)
        if not token:
            logger.error("Failed to get token. Aborting.")
            return

        while True:
            logger.info("Starting filter cycle")
            await filter_wall(session, token)
            logger.info("Filter cycle completed. Waiting for next cycle.")
            await asyncio.sleep(config.cycle_time)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_cache()
    filter_task = asyncio.create_task(run_filter())
    yield
    # Shutdown
    filter_task.cancel()
    try:
        await filter_task
    except asyncio.CancelledError:
        pass
    await save_cache()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Group Wall Filter is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
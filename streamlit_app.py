"""
1sec-clips Dashboard
Simple Streamlit app to browse clips and trigger short rendering.
"""

import io
import json
import os
import time
import zipfile
from datetime import datetime
from urllib.parse import quote

import boto3
import requests
import streamlit as st


# =============================================================================
# POSTFORME API FUNCTIONS
# =============================================================================

def get_postforme_headers():
    """Get headers for Postforme API requests."""
    api_key = st.secrets.get("postforme", {}).get("api_key", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


def postforme_get_accounts():
    """Get list of connected social accounts from Postforme."""
    try:
        response = requests.get(
            "https://api.postforme.dev/v1/social-accounts",
            headers=get_postforme_headers(),
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("data", [])
        return []
    except Exception as e:
        st.error(f"Postforme API error: {e}")
        return []


def postforme_get_auth_url(platform: str) -> str | None:
    """Get OAuth URL to connect a new social account."""
    try:
        response = requests.post(
            "https://api.postforme.dev/v1/social-accounts/auth-url",
            headers=get_postforme_headers(),
            json={"platform": platform},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("url")
        st.error(f"Failed to get auth URL: {response.text}")
        return None
    except Exception as e:
        st.error(f"Postforme API error: {e}")
        return None


def postforme_disconnect_account(account_id: str) -> bool:
    """Disconnect a social account."""
    try:
        response = requests.delete(
            f"https://api.postforme.dev/v1/social-accounts/{account_id}",
            headers=get_postforme_headers(),
            timeout=10
        )
        return response.status_code in [200, 204]
    except Exception:
        return False


def postforme_create_post(media_url: str, caption: str, account_ids: list[str], scheduled_at: str = None) -> dict:
    """Create a post on connected social accounts."""
    try:
        payload = {
            "caption": caption,
            "media": [{"url": media_url}],
            "social_accounts": account_ids,
        }
        if scheduled_at:
            payload["scheduled_at"] = scheduled_at
        
        response = requests.post(
            "https://api.postforme.dev/v1/social-posts",
            headers=get_postforme_headers(),
            json=payload,
            timeout=30
        )
        return {"success": response.status_code in [200, 201], "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Page config
st.set_page_config(
    page_title="1sec-clips Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS
st.markdown("""
<style>
    .score-badge {
        background: #ff4b4b;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .tag {
        background: #f0f2f6;
        color: #333;
        padding: 4px 8px;
        border-radius: 4px;
        margin-right: 4px;
        font-size: 12px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def get_s3_client():
    """Create S3 client from secrets."""
    return boto3.client(
        's3',
        endpoint_url=st.secrets["s3"]["endpoint"],
        aws_access_key_id=st.secrets["s3"]["access_key"],
        aws_secret_access_key=st.secrets["s3"]["secret_key"],
    )


def get_s3_public_url():
    """Get public URL base for S3."""
    endpoint = st.secrets["s3"]["endpoint"]
    bucket = st.secrets["s3"]["bucket"]
    # Convert endpoint to public URL format
    if "s3." in endpoint:
        # Format: https://s3.region.provider.com -> https://bucket.s3.region.provider.com
        return endpoint.replace("https://", f"https://{bucket}.")
    return f"{endpoint}/{bucket}"


def load_index_nocache():
    """Load index.json from S3 (no cache)."""
    try:
        s3 = get_s3_client()
        bucket = st.secrets["s3"]["bucket"]
        response = s3.get_object(Bucket=bucket, Key="index.json")
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        return {"videos": {}}


@st.cache_data(ttl=60)
def load_index():
    """Load index.json from S3."""
    return load_index_nocache()


def delete_video(video_id: str) -> tuple[bool, str]:
    """
    Delete a video and all its files from S3.
    
    Returns (success, message)
    """
    try:
        s3 = get_s3_client()
        bucket = st.secrets["s3"]["bucket"]
        prefix = f"videos/{video_id}/"
        
        # List all objects with this prefix
        objects_to_delete = []
        paginator = s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                objects_to_delete.append({'Key': obj['Key']})
        
        if not objects_to_delete:
            return False, f"No files found for video {video_id}"
        
        # Delete all objects
        s3.delete_objects(
            Bucket=bucket,
            Delete={'Objects': objects_to_delete}
        )
        
        # Update index.json
        index = load_index_nocache()
        if video_id in index.get("videos", {}):
            del index["videos"][video_id]
            s3.put_object(
                Bucket=bucket,
                Key="index.json",
                Body=json.dumps(index, ensure_ascii=False, indent=2),
                ContentType="application/json",
                ACL="public-read",
            )
        
        return True, f"Deleted {len(objects_to_delete)} files"
        
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=60)
def load_video_metadata(video_id: str):
    """Load metadata.json for a specific video."""
    try:
        s3 = get_s3_client()
        bucket = st.secrets["s3"]["bucket"]
        key = f"videos/{video_id}/metadata.json"
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        st.error(f"Failed to load metadata for {video_id}: {e}")
        return None


def trigger_github_action(event_type: str, payload: dict):
    """Trigger a GitHub Actions workflow via repository dispatch."""
    token = st.secrets.get("github", {}).get("token")
    if not token:
        st.error("GitHub token not configured")
        return False
    
    repo = st.secrets.get("github", {}).get("repo", "meanapes/1sec-clips")
    
    response = requests.post(
        f"https://api.github.com/repos/{repo}/dispatches",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
        json={
            "event_type": event_type,
            "client_payload": payload,
        }
    )
    
    return response.status_code == 204


def trigger_workflow_dispatch(workflow_file: str, inputs: dict):
    """Trigger a GitHub Actions workflow via workflow dispatch."""
    token = st.secrets.get("github", {}).get("token")
    if not token:
        st.error("GitHub token not configured")
        return False, "GitHub token not configured"
    
    repo = st.secrets.get("github", {}).get("repo", "meanapes/1sec-clips")
    
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
        json={
            "ref": "main",
            "inputs": inputs,
        }
    )
    
    if response.status_code == 204:
        return True, "Workflow triggered successfully!"
    else:
        return False, f"Error {response.status_code}: {response.text}"


def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def load_render_status(video_id: str) -> dict:
    """Load render status for a video from S3."""
    try:
        s3 = get_s3_client()
        bucket = st.secrets["s3"]["bucket"]
        # Don't URL-encode - use raw video_id to match workflow
        key = f"videos/{video_id}/render_status.json"
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception:
        return {"clips": {}}


def save_render_status(video_id: str, status: dict):
    """Save render status for a video to S3."""
    try:
        s3 = get_s3_client()
        bucket = st.secrets["s3"]["bucket"]
        # Don't URL-encode - use raw video_id to match workflow
        key = f"videos/{video_id}/render_status.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(status, ensure_ascii=False, indent=2),
            ContentType='application/json'
        )
        return True
    except Exception as e:
        st.error(f"Failed to save render status: {e}")
        return False


def check_short_exists(video_id: str, clip_id: str, s3_base_url: str) -> bool:
    """Check if a rendered short exists in S3."""
    try:
        s3 = get_s3_client()
        bucket = st.secrets["s3"]["bucket"]
        # Don't URL-encode - use raw video_id to match workflow
        key = f"videos/{video_id}/shorts/short_{clip_id}_episode.mp4"
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def create_shorts_zip(video_id: str, clips: list) -> bytes | None:
    """
    Download all ready shorts and create a ZIP archive.
    Returns ZIP file as bytes, or None if no shorts available.
    """
    s3 = get_s3_client()
    bucket = st.secrets["s3"]["bucket"]
    
    zip_buffer = io.BytesIO()
    shorts_found = 0
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for clip in clips:
            clip_id = clip.get("id", "unknown")
            key = f"videos/{video_id}/shorts/short_{clip_id}_episode.mp4"
            
            try:
                # Download from S3
                response = s3.get_object(Bucket=bucket, Key=key)
                video_data = response['Body'].read()
                
                # Add to ZIP
                zip_file.writestr(f"short_{clip_id}.mp4", video_data)
                shorts_found += 1
            except Exception:
                # Short doesn't exist yet, skip
                continue
    
    if shorts_found == 0:
        return None
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def trigger_batch_render(video_id: str, clips: list, theme: str = "episode") -> tuple[int, int]:
    """
    Trigger rendering for multiple clips at once.
    Returns (success_count, total_count)
    """
    success_count = 0
    total = len(clips)
    
    # Load or create render status
    render_status = load_render_status(video_id)
    
    for clip in clips:
        clip_id = clip.get("id", "unknown")
        
        # Update status to "queued"
        render_status["clips"][clip_id] = {
            "status": "queued",
            "queued_at": datetime.utcnow().isoformat(),
            "theme": theme,
        }
        
        # Trigger workflow
        success, _ = trigger_workflow_dispatch(
            "render-short.yml",
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "theme": theme,
            }
        )
        
        if success:
            success_count += 1
            render_status["clips"][clip_id]["status"] = "rendering"
            render_status["clips"][clip_id]["started_at"] = datetime.utcnow().isoformat()
        else:
            render_status["clips"][clip_id]["status"] = "failed"
            render_status["clips"][clip_id]["error"] = "Failed to trigger workflow"
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Save status
    save_render_status(video_id, render_status)
    
    return success_count, total


def render_clip_card(clip: dict, video_id: str, s3_base_url: str):
    """Render a single clip card."""
    clip_id = clip.get("id", "unknown")
    # URL-encode video_id for Cyrillic characters
    encoded_video_id = quote(video_id, safe='')
    clip_url = f"{s3_base_url}/videos/{encoded_video_id}/clips/{clip_id}.mp4"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.video(clip_url)
    
    with col2:
        # Header with ID and score
        # Support both nested "moment" structure and flat structure
        moment = clip.get("moment", clip)
        score = moment.get("virality_score", clip.get("virality_score", 0))
        
        st.markdown(f"### {clip_id} &nbsp; <span class='score-badge'>{score:.1f}</span>", unsafe_allow_html=True)
        
        # Transcript
        transcript = moment.get("transcript", clip.get("transcript", "No transcript"))
        st.markdown(f"**Transcript:** {transcript[:200]}{'...' if len(transcript) > 200 else ''}")
        
        # Timing
        start = clip.get("start", 0)
        end = clip.get("end", 0)
        duration = clip.get("duration", end - start)
        st.markdown(f"**Timing:** {format_duration(start)} → {format_duration(end)} ({duration:.1f}s)")
        
        # Type and mood
        clip_type = moment.get("type", clip.get("type", "unknown"))
        mood = moment.get("mood", clip.get("mood", "neutral"))
        st.markdown(f"**Type:** {clip_type} · **Mood:** {mood}")
        
        # Tags
        tags = moment.get("tags", clip.get("tags", []))
        if tags:
            tags_html = " ".join([f"<span class='tag'>#{tag}</span>" for tag in tags[:6]])
            st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)
        
        # Characters
        characters = moment.get("characters", clip.get("characters", []))
        if characters:
            st.markdown(f"**Characters:** {', '.join(characters)}")
        
        # Reason
        # Prefer Russian reason, fallback to English
        reason_ru = moment.get("reason_ru", clip.get("reason_ru", ""))
        reason = reason_ru or moment.get("reason", clip.get("reason", ""))
        if reason:
            st.markdown(f"**Почему:** {reason}")
        
        # Check if short exists
        encoded_video_id = quote(video_id, safe='')
        short_url_episode = f"{s3_base_url}/videos/{encoded_video_id}/shorts/short_{clip_id}_episode.mp4"
        
        # Actions
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.link_button("⬇️ Clip", clip_url, use_container_width=True)
        with col_b:
            if st.button("🎬 Render Short", key=f"short_{video_id}_{clip_id}", use_container_width=True):
                st.session_state[f"render_short_{video_id}_{clip_id}"] = True
        with col_c:
            st.link_button("📱 Short", short_url_episode, use_container_width=True, help="Download rendered short (if exists)")
        
        # Show render options if button was clicked
        if st.session_state.get(f"render_short_{video_id}_{clip_id}"):
            with st.form(key=f"short_form_{video_id}_{clip_id}"):
                st.subheader("🎬 Render Short")
                theme = st.selectbox("Theme", ["episode", "minimal", "cinematic"], key=f"theme_{video_id}_{clip_id}")
                
                if st.form_submit_button("🚀 Start Rendering"):
                    success, message = trigger_workflow_dispatch(
                        "render-short.yml",
                        {
                            "video_id": video_id,
                            "clip_id": clip_id,
                            "theme": theme,
                        }
                    )
                    if success:
                        st.success("✅ Rendering started!")
                        st.session_state[f"render_short_{video_id}_{clip_id}"] = False
                    else:
                        st.error(f"❌ {message}")
            
            # Show GitHub Actions link outside the form (always visible after triggering)
            st.link_button(
                "👁️ Смотреть в GitHub Actions",
                f"https://github.com/{st.secrets.get('github', {}).get('repo', 'meanapes/1sec-clips')}/actions/workflows/render-short.yml",
                use_container_width=True
            )
    
    st.divider()


def main():
    # Sidebar
    with st.sidebar:
        st.title("🎬 1sec-clips")
        st.markdown("Browse and manage video clips")
        
        # Load index
        index = load_index()
        videos = index.get("videos", {})
        
        if not videos:
            st.warning("No videos found yet")
            # Still show process new video section
            st.divider()
            st.subheader("Actions")
            with st.expander("🎥 Process New Video", expanded=True):
                video_url = st.text_input(
                    "Video URL",
                    placeholder="YouTube URL or magnet link",
                    help="Paste a YouTube video URL or magnet link",
                    key="empty_video_url"
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    proc_min_score = st.number_input("Min Score", 0.0, 10.0, 6.0, 0.5, key="empty_min_score")
                with col_b:
                    proc_max_duration = st.number_input("Max Duration (s)", 20, 300, 180, 10, key="empty_max_dur")
                
                col_c, col_d = st.columns(2)
                with col_c:
                    proc_padding = st.number_input("Padding (s)", 0, 30, 3, 1, key="empty_padding", help="Секунды до/после момента")
                with col_d:
                    audio_lang = st.selectbox(
                        "Audio Language",
                        ["ru", "en", "auto"],
                        index=0,
                        key="empty_audio_lang",
                        help="Prefer Russian audio track"
                    )
                
                # Fixed to gemini-3-pro-preview (best quality)
                ai_model = "gemini-3-pro-preview"
                st.text_input("AI Model", value=ai_model, disabled=True, key="empty_ai_model")
                
                if st.button("🚀 Start Processing", type="primary", disabled=not video_url, key="empty_process"):
                    with st.spinner("Triggering workflow..."):
                        success, message = trigger_workflow_dispatch(
                            "process-video.yml",
                            {
                                "video_url": video_url,
                                "min_score": str(proc_min_score),
                                "max_duration": str(proc_max_duration),
                                "model": ai_model,
                                "padding": str(proc_padding),
                                "audio_language": audio_lang,
                            }
                        )
                        if success:
                            st.success("✅ Processing started! Check GitHub Actions for progress.")
                            st.link_button(
                                "View GitHub Actions",
                                f"https://github.com/{st.secrets.get('github', {}).get('repo', 'meanapes/1sec-clips')}/actions"
                            )
                        else:
                            st.error(f"❌ {message}")
            st.stop()
        
        # Video selector
        video_options = {
            vid: data.get("title", vid)[:40]
            for vid, data in videos.items()
        }
        
        selected_video = st.selectbox(
            "Select Video",
            options=list(video_options.keys()),
            format_func=lambda x: video_options[x],
        )
        
        # Video actions
        if selected_video:
            # Re-process button
            with st.expander("🔄 Перезалив", expanded=False):
                # Load metadata to get original URL
                video_meta = load_video_metadata(selected_video)
                original_url = video_meta.get("original_url", "") if video_meta else ""
                
                if original_url:
                    st.info(f"Перезапустить обработку с теми же или новыми настройками")
                    # Show full URL in expandable text area (not truncated!)
                    with st.expander("📋 Исходный URL", expanded=False):
                        st.text_area("URL", value=original_url, height=100, disabled=True, key="reprocess_url_full")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        reprocess_min_score = st.number_input("Min Score", 0.0, 10.0, 6.0, 0.5, key="reprocess_min_score")
                    with col_b:
                        reprocess_max_duration = st.number_input("Max Duration (s)", 20, 300, 180, 10, key="reprocess_max_dur")
                    
                    col_c, col_d = st.columns(2)
                    with col_c:
                        reprocess_padding = st.number_input("Padding (s)", 0, 30, 3, 1, key="reprocess_padding", help="Секунды до/после момента")
                    with col_d:
                        reprocess_audio_lang = st.selectbox(
                            "Audio Language",
                            ["ru", "en", "auto"],
                            index=0,
                            key="reprocess_audio_lang"
                        )
                    
                    # Fixed to gemini-3-pro-preview (best quality)
                    reprocess_model = "gemini-3-pro-preview"
                    st.text_input("AI Model", value=reprocess_model, disabled=True, key="reprocess_model")
                    
                    if st.button("🚀 Запустить перезалив", type="primary", key="reprocess_btn"):
                        with st.spinner("Запускаю обработку..."):
                            success, message = trigger_workflow_dispatch(
                                "process-video.yml",
                                {
                                    "video_url": original_url,
                                    "min_score": str(reprocess_min_score),
                                    "max_duration": str(reprocess_max_duration),
                                    "model": reprocess_model,
                                    "padding": str(reprocess_padding),
                                    "audio_language": reprocess_audio_lang,
                                }
                            )
                            if success:
                                st.success("✅ Обработка запущена!")
                                st.link_button(
                                    "Смотреть в GitHub Actions",
                                    f"https://github.com/{st.secrets.get('github', {}).get('repo', 'meanapes/1sec-clips')}/actions"
                                )
                            else:
                                st.error(f"❌ {message}")
                else:
                    st.warning("URL источника не найден в метаданных")
            
            # Delete video button
            with st.expander("🗑️ Delete Video", expanded=False):
                st.warning(f"This will permanently delete **{video_options[selected_video]}** and all its clips!")
                
                confirm_text = st.text_input(
                    "Type 'DELETE' to confirm",
                    key="delete_confirm"
                )
                
                if st.button("🗑️ Delete Permanently", type="primary", disabled=confirm_text != "DELETE"):
                    with st.spinner("Deleting..."):
                        success, message = delete_video(selected_video)
                        if success:
                            st.success(f"✅ {message}")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
        
        st.divider()
        
        # Filters
        st.subheader("Filters")
        
        min_score = st.slider("Min Score", 0.0, 10.0, 0.0, 0.5)
        
        clip_types = ["all", "dialogue", "funny", "emotional", "quote", "plot_twist", "iconic", "insight"]
        selected_type = st.selectbox("Clip Type", clip_types)
        
        st.divider()
        
        # Actions
        st.subheader("Actions")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Process New Video
        with st.expander("🎥 Process New Video"):
            video_url = st.text_input(
                "Video URL",
                placeholder="YouTube URL or magnet link",
                help="Paste a YouTube video URL or magnet link"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                proc_min_score = st.number_input("Min Score", 0.0, 10.0, 6.0, 0.5)
            with col_b:
                proc_max_duration = st.number_input("Max Duration (s)", 20, 300, 180, 10)
            
            col_c, col_d = st.columns(2)
            with col_c:
                proc_padding = st.number_input("Padding (s)", 0, 30, 3, 1, help="Секунды до/после момента")
            with col_d:
                audio_lang = st.selectbox(
                    "Audio Language",
                    ["ru", "en", "auto"],
                    index=0,
                    help="Prefer Russian audio track (for movies with multiple dubs)"
                )
            
            # Fixed to gemini-3-pro-preview (best quality)
            ai_model = "gemini-3-pro-preview"
            st.text_input("AI Model", value=ai_model, disabled=True)
            
            if st.button("🚀 Start Processing", type="primary", disabled=not video_url):
                with st.spinner("Triggering workflow..."):
                    success, message = trigger_workflow_dispatch(
                        "process-video.yml",
                        {
                            "video_url": video_url,
                            "min_score": str(proc_min_score),
                            "max_duration": str(proc_max_duration),
                            "model": ai_model,
                            "padding": str(proc_padding),
                            "audio_language": audio_lang,
                        }
                    )
                    if success:
                        st.success("✅ Processing started! Check GitHub Actions for progress.")
                        st.link_button(
                            "View GitHub Actions",
                            f"https://github.com/{st.secrets.get('github', {}).get('repo', 'meanapes/1sec-clips')}/actions"
                        )
                    else:
                        st.error(f"❌ {message}")
    
    # Main content
    if selected_video:
        metadata = load_video_metadata(selected_video)
        
        if not metadata:
            st.error("Failed to load video metadata")
            st.stop()
        
        # Video header
        title = metadata.get("title", selected_video)
        st.title(title)
        
        # Video info
        movie_info = metadata.get("movie_info", {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clips", len(metadata.get("clips", [])))
        with col2:
            duration = metadata.get("duration_seconds", 0)
            st.metric("Duration", format_duration(duration))
        with col3:
            st.metric("Source", metadata.get("source", "unknown"))
        with col4:
            processed = metadata.get("processed_at", "")[:10]
            st.metric("Processed", processed)
        
        # Movie info if available
        if movie_info.get("year") or movie_info.get("genres"):
            with st.expander("📽️ Movie Info"):
                if movie_info.get("year"):
                    st.write(f"**Year:** {movie_info['year']}")
                if movie_info.get("genres"):
                    st.write(f"**Genres:** {', '.join(movie_info['genres'])}")
                if movie_info.get("synopsis"):
                    st.write(f"**Synopsis:** {movie_info['synopsis']}")
                if movie_info.get("cast"):
                    cast_names = [c.get("name", "") for c in movie_info["cast"][:5]]
                    st.write(f"**Cast:** {', '.join(cast_names)}")
        
        st.divider()
        
        # Tabs: Clips | Gallery | Publish
        tab_clips, tab_gallery, tab_publish = st.tabs(["📋 Clips", "🎬 Gallery", "📤 Publish"])
        
        # Clips
        clips = metadata.get("clips", [])
        s3_base_url = get_s3_public_url()
        
        with tab_clips:
            # Apply filters
            filtered_clips = []
            for clip in clips:
                moment = clip.get("moment", clip)
                score = moment.get("virality_score", clip.get("virality_score", 0))
                clip_type = moment.get("type", clip.get("type", "unknown"))
                
                if score < min_score:
                    continue
                if selected_type != "all" and clip_type != selected_type:
                    continue
                
                filtered_clips.append(clip)
            
            # Sort by score
            def get_score(c):
                m = c.get("moment", c)
                return m.get("virality_score", c.get("virality_score", 0))
            filtered_clips.sort(key=get_score, reverse=True)
            
            # Batch render section
            col_info, col_action = st.columns([2, 1])
            with col_info:
                st.subheader(f"Clips ({len(filtered_clips)} of {len(clips)})")
            with col_action:
                if st.button("🎬 Render All Shorts", type="primary", use_container_width=True):
                    st.session_state["show_batch_render"] = True
            
            # Batch render confirmation
            if st.session_state.get("show_batch_render"):
                with st.container():
                    st.info(f"🎬 **Render {len(filtered_clips)} shorts?**")
                    st.caption(f"Estimated cost: ~€{len(filtered_clips) * 0.5 * 0.022:.2f} (avg 30s per short)")
                    
                    col_theme, col_confirm, col_cancel = st.columns([2, 1, 1])
                    with col_theme:
                        batch_theme = st.selectbox("Theme", ["episode", "minimal", "cinematic"], key="batch_theme")
                    with col_confirm:
                        if st.button("✅ Start All", type="primary", use_container_width=True):
                            with st.spinner(f"Triggering {len(filtered_clips)} render jobs..."):
                                success, total = trigger_batch_render(selected_video, filtered_clips, batch_theme)
                            st.success(f"✅ Started {success}/{total} render jobs!")
                            st.session_state["show_batch_render"] = False
                            st.link_button(
                                "👁️ View in GitHub Actions",
                                f"https://github.com/{st.secrets.get('github', {}).get('repo', 'meanapes/1sec-clips')}/actions/workflows/render-short.yml",
                                use_container_width=True
                            )
                    with col_cancel:
                        if st.button("❌ Cancel", use_container_width=True):
                            st.session_state["show_batch_render"] = False
                            st.rerun()
            
            st.divider()
            
            # Render clips
            for clip in filtered_clips:
                render_clip_card(clip, selected_video, s3_base_url)
        
        with tab_gallery:
            # Header with refresh button
            col_title, col_refresh = st.columns([3, 1])
            with col_title:
                st.subheader("🎬 Rendered Shorts")
            with col_refresh:
                if st.button("🔄 Refresh", key="gallery_refresh_btn", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            
            # Load render status
            render_status = load_render_status(selected_video)
            
            # Count stats
            ready_count = 0
            rendering_count = 0
            queued_count = 0
            
            for clip in clips:
                clip_id = clip.get("id", "unknown")
                # Check if short actually exists in S3
                if check_short_exists(selected_video, clip_id, s3_base_url):
                    ready_count += 1
                elif clip_id in render_status.get("clips", {}):
                    status = render_status["clips"][clip_id].get("status", "unknown")
                    if status == "rendering":
                        rendering_count += 1
                    elif status == "queued":
                        queued_count += 1
            
            # Stats row
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("✅ Ready", ready_count)
            with col_s2:
                st.metric("⏳ Rendering", rendering_count)
            with col_s3:
                st.metric("📋 Queued", queued_count)
            with col_s4:
                st.metric("📊 Total", len(clips))
            
            # Download All button
            if ready_count > 0:
                col_dl1, col_dl2 = st.columns([1, 3])
                with col_dl1:
                    if st.button("📦 Скачать всё (ZIP)", key="download_all_zip", use_container_width=True):
                        with st.spinner(f"Создаю архив ({ready_count} шортсов)..."):
                            zip_data = create_shorts_zip(selected_video, clips)
                            if zip_data:
                                st.session_state["zip_data"] = zip_data
                                st.session_state["zip_ready"] = True
                            else:
                                st.error("Не удалось создать архив")
                
                # Show download button if ZIP is ready
                if st.session_state.get("zip_ready"):
                    with col_dl1:
                        # Clean filename from video title
                        safe_name = "".join(c for c in selected_video if c.isalnum() or c in (' ', '-', '_')).strip()[:30]
                        st.download_button(
                            label="⬇️ Скачать ZIP",
                            data=st.session_state["zip_data"],
                            file_name=f"{safe_name}_shorts.zip",
                            mime="application/zip",
                            key="download_zip_btn",
                            use_container_width=True
                        )
            
            st.divider()
            
            # Gallery grid
            cols_per_row = 3
            rows = [clips[i:i + cols_per_row] for i in range(0, len(clips), cols_per_row)]
            
            for row in rows:
                cols = st.columns(cols_per_row)
                for idx, clip in enumerate(row):
                    with cols[idx]:
                        clip_id = clip.get("id", "unknown")
                        moment = clip.get("moment", clip)
                        encoded_video_id = quote(selected_video, safe='')
                        short_url = f"{s3_base_url}/videos/{encoded_video_id}/shorts/short_{clip_id}_episode.mp4"
                        
                        # Check status
                        exists = check_short_exists(selected_video, clip_id, s3_base_url)
                        clip_status = render_status.get("clips", {}).get(clip_id, {})
                        
                        # Status indicator
                        if exists:
                            st.success(f"✅ {clip_id}")
                            st.video(short_url)
                            st.link_button("⬇️ Download", short_url, use_container_width=True)
                        elif clip_status.get("status") == "rendering":
                            st.warning(f"⏳ {clip_id} - Rendering...")
                            started = clip_status.get("started_at", "")[:19]
                            st.caption(f"Started: {started}")
                        elif clip_status.get("status") == "queued":
                            st.info(f"📋 {clip_id} - Queued")
                        elif clip_status.get("status") == "failed":
                            st.error(f"❌ {clip_id} - Failed")
                            st.caption(clip_status.get("error", "Unknown error"))
                        else:
                            st.markdown(f"⬜ {clip_id}")
                            # Quick render button
                            if st.button("🎬 Render", key=f"gallery_render_{clip_id}", use_container_width=True):
                                success, msg = trigger_workflow_dispatch(
                                    "render-short.yml",
                                    {"video_id": selected_video, "clip_id": clip_id, "theme": "episode"}
                                )
                                if success:
                                    st.success("Started!")
                                    # Update status
                                    render_status.setdefault("clips", {})[clip_id] = {
                                        "status": "rendering",
                                        "started_at": datetime.utcnow().isoformat(),
                                    }
                                    save_render_status(selected_video, render_status)
                                    st.rerun()
                        
                        # Show description
                        reason_ru = moment.get("reason_ru", "")
                        if reason_ru:
                            st.caption(reason_ru[:50] + "..." if len(reason_ru) > 50 else reason_ru)
        
        # =================================================================
        # PUBLISH TAB
        # =================================================================
        with tab_publish:
            st.subheader("📤 Publish to Social Media")
            
            # Check if Postforme API key is configured
            postforme_key = st.secrets.get("postforme", {}).get("api_key", "")
            if not postforme_key:
                st.warning("⚠️ Postforme API key not configured. Add it to secrets as `postforme.api_key`")
                st.stop()
            
            # Connected accounts section
            st.markdown("### 🔗 Подключённые аккаунты")
            
            accounts = postforme_get_accounts()
            
            if accounts:
                for acc in accounts:
                    col_acc, col_action = st.columns([4, 1])
                    with col_acc:
                        platform = acc.get("platform", "unknown")
                        name = acc.get("name", acc.get("username", "Unknown"))
                        platform_icons = {
                            "youtube": "▶️",
                            "tiktok": "🎵",
                            "instagram": "📸",
                            "twitter": "🐦",
                            "x": "𝕏",
                        }
                        icon = platform_icons.get(platform.lower(), "🔗")
                        st.markdown(f"{icon} **{platform.title()}**: {name}")
                    with col_action:
                        if st.button("🗑️", key=f"disconnect_{acc.get('id')}", help="Отключить"):
                            if postforme_disconnect_account(acc.get("id")):
                                st.success("Отключено!")
                                st.rerun()
                            else:
                                st.error("Ошибка")
            else:
                st.info("Нет подключённых аккаунтов")
            
            st.divider()
            
            # Connect new account
            st.markdown("### ➕ Подключить новый аккаунт")
            
            col_yt, col_tt, col_ig, col_tw = st.columns(4)
            
            with col_yt:
                if st.button("▶️ YouTube", use_container_width=True):
                    auth_url = postforme_get_auth_url("youtube")
                    if auth_url:
                        st.markdown(f"[🔗 Открыть авторизацию YouTube]({auth_url})")
                        st.info("После авторизации обнови страницу")
            
            with col_tt:
                if st.button("🎵 TikTok", use_container_width=True):
                    auth_url = postforme_get_auth_url("tiktok")
                    if auth_url:
                        st.markdown(f"[🔗 Открыть авторизацию TikTok]({auth_url})")
                        st.info("После авторизации обнови страницу")
            
            with col_ig:
                if st.button("📸 Instagram", use_container_width=True):
                    auth_url = postforme_get_auth_url("instagram")
                    if auth_url:
                        st.markdown(f"[🔗 Открыть авторизацию Instagram]({auth_url})")
                        st.info("После авторизации обнови страницу")
            
            with col_tw:
                if st.button("𝕏 Twitter", use_container_width=True):
                    auth_url = postforme_get_auth_url("twitter")
                    if auth_url:
                        st.markdown(f"[🔗 Открыть авторизацию Twitter]({auth_url})")
                        st.info("После авторизации обнови страницу")
            
            st.divider()
            
            # Publish shorts section
            st.markdown("### 🎬 Опубликовать шортсы")
            
            if not accounts:
                st.warning("Сначала подключи хотя бы один аккаунт")
            else:
                # Get ready shorts
                ready_shorts = []
                for clip in clips:
                    clip_id = clip.get("id", "unknown")
                    if check_short_exists(selected_video, clip_id, s3_base_url):
                        moment = clip.get("moment", clip)
                        encoded_video_id = quote(selected_video, safe='')
                        short_url = f"{s3_base_url}/videos/{encoded_video_id}/shorts/short_{clip_id}_episode.mp4"
                        ready_shorts.append({
                            "clip_id": clip_id,
                            "url": short_url,
                            "reason_ru": moment.get("reason_ru", f"Момент из {selected_video}"),
                        })
                
                if not ready_shorts:
                    st.info("Нет готовых шортсов для публикации. Сначала отрендери их во вкладке Gallery.")
                else:
                    st.success(f"✅ {len(ready_shorts)} шортсов готово к публикации")
                    
                    # Select shorts to publish
                    selected_shorts = st.multiselect(
                        "Выбери шортсы для публикации",
                        options=[s["clip_id"] for s in ready_shorts],
                        format_func=lambda x: f"{x}: {next((s['reason_ru'][:40] for s in ready_shorts if s['clip_id'] == x), '')}..."
                    )
                    
                    # Select accounts to publish to
                    selected_accounts = st.multiselect(
                        "Куда публиковать",
                        options=[acc.get("id") for acc in accounts],
                        format_func=lambda x: f"{next((acc.get('platform', '').title() + ': ' + acc.get('name', '') for acc in accounts if acc.get('id') == x), x)}"
                    )
                    
                    # Caption template
                    caption_template = st.text_area(
                        "Шаблон подписи",
                        value="{reason}\n\n#shorts #кино #фильмы",
                        help="Используй {reason} для автоподстановки описания момента"
                    )
                    
                    # Publish button
                    if st.button("🚀 Опубликовать", type="primary", disabled=not selected_shorts or not selected_accounts):
                        progress = st.progress(0)
                        for i, clip_id in enumerate(selected_shorts):
                            short = next((s for s in ready_shorts if s["clip_id"] == clip_id), None)
                            if short:
                                caption = caption_template.replace("{reason}", short["reason_ru"])
                                result = postforme_create_post(
                                    media_url=short["url"],
                                    caption=caption,
                                    account_ids=selected_accounts
                                )
                                if result["success"]:
                                    st.success(f"✅ {clip_id} опубликован!")
                                else:
                                    st.error(f"❌ {clip_id}: {result.get('error', 'Unknown error')}")
                            progress.progress((i + 1) / len(selected_shorts))
                        st.balloons()


if __name__ == "__main__":
    main()

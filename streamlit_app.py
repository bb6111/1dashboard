"""
1sec-clips Dashboard
Simple Streamlit app to browse clips and trigger short rendering.
"""

import json
import os
from datetime import datetime
from urllib.parse import quote

import boto3
import requests
import streamlit as st

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


@st.cache_data(ttl=60)
def load_index():
    """Load index.json from S3."""
    try:
        s3 = get_s3_client()
        bucket = st.secrets["s3"]["bucket"]
        response = s3.get_object(Bucket=bucket, Key="index.json")
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return {"videos": {}}


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
        reason = moment.get("reason", clip.get("reason", ""))
        if reason:
            st.markdown(f"**Why:** {reason}")
        
        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            st.link_button("⬇️ Download", clip_url)
        
        with btn_col2:
            if st.button("🎬 Make Short", key=f"short_{video_id}_{clip_id}"):
                with st.spinner("Triggering render..."):
                    success = trigger_github_action("render_short", {
                        "video_id": video_id,
                        "clip_id": clip_id,
                    })
                    if success:
                        st.success("Render triggered! Check GitHub Actions.")
                    else:
                        st.error("Failed to trigger render")
        
        with btn_col3:
            # Copy timestamp button
            st.code(f"{format_duration(start)} - {format_duration(end)}")
    
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
                
                ai_model = st.selectbox(
                    "AI Model",
                    ["gemini-2.0-flash", "gemini-3-flash-preview", "gemini-3-pro-preview"],
                    index=0,
                    key="empty_ai_model"
                )
                
                audio_lang = st.selectbox(
                    "Audio Language",
                    ["ru", "en", "auto"],
                    index=0,
                    key="empty_audio_lang",
                    help="Prefer Russian audio track (for movies with multiple dubs)"
                )
                
                if st.button("🚀 Start Processing", type="primary", disabled=not video_url, key="empty_process"):
                    with st.spinner("Triggering workflow..."):
                        success, message = trigger_workflow_dispatch(
                            "process-video.yml",
                            {
                                "video_url": video_url,
                                "min_score": str(proc_min_score),
                                "max_duration": str(proc_max_duration),
                                "model": ai_model,
                                "audio_language": audio_lang,
                            }
                        )
                        if success:
                            st.success("✅ Processing started! Check GitHub Actions for progress.")
                            st.link_button(
                                "View GitHub Actions",
                                f"https://github.com/{st.secrets.get('github', {}).get('repo', 'meanapes/1sec-clips')}/actions",
                                key="empty_gh_link"
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
            
            ai_model = st.selectbox(
                "AI Model",
                ["gemini-2.0-flash", "gemini-3-flash-preview", "gemini-3-pro-preview"],
                index=0
            )
            
            audio_lang = st.selectbox(
                "Audio Language",
                ["ru", "en", "auto"],
                index=0,
                help="Prefer Russian audio track (for movies with multiple dubs)"
            )
            
            if st.button("🚀 Start Processing", type="primary", disabled=not video_url):
                with st.spinner("Triggering workflow..."):
                    success, message = trigger_workflow_dispatch(
                        "process-video.yml",
                        {
                            "video_url": video_url,
                            "min_score": str(proc_min_score),
                            "max_duration": str(proc_max_duration),
                            "model": ai_model,
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
        
        # Clips
        clips = metadata.get("clips", [])
        
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
        
        st.subheader(f"Clips ({len(filtered_clips)} of {len(clips)})")
        
        # Sort by score
        def get_score(c):
            m = c.get("moment", c)
            return m.get("virality_score", c.get("virality_score", 0))
        filtered_clips.sort(key=get_score, reverse=True)
        
        # Render clips
        s3_base_url = get_s3_public_url()
        
        for clip in filtered_clips:
            render_clip_card(clip, selected_video, s3_base_url)


if __name__ == "__main__":
    main()

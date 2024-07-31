import os
import time
import requests
from yt_dlp import YoutubeDL

from pathlib import Path
from typing import List
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile, APIRouter, Request, Query, Depends, HTTPException
router = APIRouter(
    prefix="/reasoning",
    tags=['Adaptive Feedback (QnA reasoning)']
)

@router.post("/convert2kb/youtube-url", summary="Extract knowledge from a YouTube video URL to KB")
async def convert_youtube_url_to_kb(youtube_url: str):
    print(f"==>[INFO] try to start : {youtube_url}")
    # Download the YouTube video
    ydl_opts = {
      'verbose': False,
      'quiet': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        video_id = info_dict.get("id", None)
        if not video_id:
            raise HTTPException(status_code=400, detail="Failed to extract video ID from URL")
        video_path = ydl.prepare_filename(info_dict)
        # video_subs = info_dict.get()

    # Process the downloaded video
    # For demonstration purposes, we'll just return the video ID and path
    return {"video_id": video_id, "video_path": video_path, "info_dict": info_dict}

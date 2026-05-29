# TODO

- [ ] Build CLI interface with all commands
- [ ] Add a project skill documenting all CLI capabilities
- [ ] Add a final post-processing step: after STT produces the transcript, clean it up and produce a final transcript that overwrites the original `.txt` on Google Drive. Extract the 2 interlocutor names from the video filename and map them to `Speaker N`. If there are extra speakers (when only 2 interlocutors are expected), decide which real speaker each extra one should be merged into.
- [ ] Fix dropped characters in output filenames: a Drive video `Yana Banova and Andrei Smirnov - 2026/05/28 17:27 GMT+04:00 – Recording.mp4` produced siblings named `28 17:27 GMT+04:00 – Recording` — `/` in the Drive name is treated as a path separator by `Path(...).stem`/`.name` (src/drive.py, src/main.py), dropping everything before the last `/`. Use `os.path.splitext` for Drive-name stems and decouple local temp filenames from uploaded Drive names.
- [ ] Add an OpenAI pipeline that works like the keypoints-transcription skill at /home/popstas/projects/text/obsidian/ExpertizeMe/.claude/skills/keypoints-transcription. Config: openai_api_key, proxy_url, model gpt-5.4-mini, prompt based on the skill. Use the modern OpenAI Responses API, maybe the OpenAI Agents SDK — agent can run the Python scripts referenced in the skill. Consider batch mode support (drops price by 50%).

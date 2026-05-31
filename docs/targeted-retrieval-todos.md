# Targeted Retrieval Todos

- Keep section labels canonical in ingestion/chunking only.
- Prefer SQLite first for candidate, skill, education, and document lookups.
- Use `section_type` only to narrow chunk fetch after SQL resolution.
- Keep full CV requests on the direct document path, not RAG.
- Fall back to hybrid retrieval when targeted SQL lookup resolves nothing.
- Add company and role targeted lookup after the current slice is stable.

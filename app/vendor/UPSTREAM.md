# Original LightOnOCR viewer

Source: https://github.com/lightonai/LightOnOCR
Pinned commit: 36755d461be079737860a5f03ae0c803501269e9
License: Apache-2.0, retained in LICENSE.lightonocr.

The original lightonocr Python package and static viewer assets are vendored here. Its server, pipeline, grounding parser, CSS, layout and interaction design are retained. The launcher wrapper supplies a local Transformers OpenAI-compatible endpoint; vLLM is neither installed nor used.

Narrow frontend modifications use locally installed pinned rendering libraries, sanitize rendered Markdown/HTML with DOMPurify, and escape model-provided labels. These are security/offline changes, not a replacement frontend. Original uploads and results persist under app/out.

Browser libraries are installed by app/frontend_assets.py from pinned npm archives verified with SHA-512 integrity values. Marked14.1.3 and KaTeX0.16.11 match upstream. DOMPurify3.4.16 adds sanitization. Their licenses are retained beside installed assets under static/lib. No npm scripts are executed.

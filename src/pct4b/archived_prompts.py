"""Archived inference prompt templates; extracted without endpoints or credentials."""


class Prompts:
    @staticmethod
    def get_subdomain_prompt(path, sub_per_instance):
        path_str = "-->".join(path)
        prompt = f'You are a taxonomy builder. Given a hierarchical field path and a target count, propose subdomains that are exactly ONE level below the final node in the path.\nINPUTS\n- path: {path_str}\n- sub_per_instance: {sub_per_instance}  // integer N\n\nRULES\n1) Output exactly N unique subdomains (no more, no fewer).\n2) Each subdomain must be a legitimate, commonly recognized branch under the given path (one level deeper only; not siblings or cross-domain).\n3) Make the set MECE: mutually exclusive, minimal overlap, broadly covering the parent scope at that level.\n4) Granularity: each item should be a concise noun phrase (2–5 words), Title Case, no punctuation, no parent term repeated.\n5) No catch-alls like “Miscellaneous,” no methods/skills unless they are standard subfields, no brand/tool names unless canonical to the domain.\n6) If the parent is already at a leaf (i.e., no reasonable subdomains exist), return an empty list [].\n7) Output format must be a pure JSON array of strings, nothing else (no prose, no keys).\n\nNOW RETURN:\nA JSON array of exactly N subdomain names for the parent "{path_str}".\n'
        return prompt

    @staticmethod
    def text_generation_from_topic(topic_path):
        topic_path_str = "-->".join(topic_path)
        prompt = f"Generate text for topic: {topic_path_str}"
        return prompt

    @staticmethod
    def text_translation_messages(text, source_language, target_language):
        system_prompt = f'You are an expert translator. Translate the following text from {source_language} into {target_language}.\n\nRequirements:\n- Output ONLY the translated text and nothing else — no explanations, headings, language names, tags, quotes, code fences, or metadata.\n- Preserve the original formatting exactly (line breaks, paragraphs, lists, bullets, indentation, tables). Only change the words to {target_language}.\n- Preserve numbers, dates, times, currencies and units as written unless a standard localized form is clearly expected in {target_language}.\n- If the source contains markup (HTML, Markdown) or code blocks, do NOT modify tags or code; translate only visible/plain text.\n- Preserve proper names, trademarks, and acronyms; do not transliterate them unless required by {target_language} conventions.\n- Match the tone and register of the source (formal → formal, informal → informal); if ambiguous, use a natural, neutral tone appropriate for native speakers of {target_language}.\n- If a word cannot be translated, leave it as-is.\n- Do not add or remove blank lines or extra whitespace.\n- If {source_language} is "auto" or "detect", detect the source language automatically.\n\nSource language: {source_language}\nTarget language: {target_language}\n\n'
        user_prompt = f"Text:\n'''{text}'''\n"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def text_translation_messages_short(text, source_language, target_language, model_name):
        if model_name == "gemma-3-4b-translate":
            system_prompt = (
                f"Translate the following text from {source_language} into {target_language}."
            )
            user_prompt = f"\n{text}"
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            system_prompt = f"You are an expert translator. Translate the following text from {source_language} into {target_language}.\n\n    Requirements:\n    - Preserve the original formatting exactly (line breaks, paragraphs, lists, bullets, indentation, tables). Only change the words to {target_language}.\n    - Preserve numbers, dates, times, currencies and units as written unless a standard localized form is clearly expected in {target_language}.\n    - If the source contains markup (HTML, Markdown) or code blocks, do NOT modify tags or code; translate only visible/plain text.\n    - Preserve proper names, trademarks, and acronyms; do not transliterate them unless required by {target_language} conventions.\n    - Match the tone and register of the source (formal → formal, informal → informal); if ambiguous, use a natural, neutral tone appropriate for native speakers of {target_language}.\n    - If a word cannot be translated, leave it as-is.\n    - Do not add or remove blank lines or extra whitespace.\n\n    Source language: {source_language}\n    Target language: {target_language}\n    "
            user_prompt = f"Text:\n    '''{text}'''\n    "
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

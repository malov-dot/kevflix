"""One owned startup MT request using the production client; no TTS or movie credit."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    record = {'kind': 'synthetic-startup', 'stage': 'mt_warmup', 'source_language': 'ru',
              'target_language': 'en', 'movie_credit': False, 'audio_generated': False,
              'output_discarded': True, 'ok': False}
    try:
        from backend.config import Settings
        from backend.services import OpenAIChatTranslateService

        settings = Settings.from_env()
        provider = (settings.translation_provider or 'hunyuan').strip().lower().replace('_', '-')
        if provider != 'lmt':
            raise ValueError('Startup warmup requires the configured LMT provider')
        client = OpenAIChatTranslateService(settings.lmt_url, provider=provider,
                                            model=settings.translation_model,
                                            max_tokens=settings.translation_max_tokens)
        try:
            output = client.translate('Да.', 'ru', 'en')
            if not isinstance(output, str) or not output.strip():
                raise ValueError('MT startup request returned empty output')
            del output
            record.update(ok=True, model=client.model, max_tokens=client.max_tokens)
        finally:
            client.session.close()
    except Exception as error:
        record.update(error_type=type(error).__name__, message=str(error))
    record['seconds'] = time.perf_counter() - started
    Path(args.receipt).write_text(json.dumps(record, sort_keys=True), encoding='utf-8')
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if record['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

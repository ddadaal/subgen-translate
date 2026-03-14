import logging
import os
import re
from threading import Lock
from typing import List, Optional

import torch
from tqdm import tqdm

from language_code import LanguageCode


translation_model = None
translation_processor = None
translation_model_lock = Lock()

_model_id = 'google/translategemma-4b-it'
_model_location = './models'
_max_new_tokens = 192
_batch_size = 4


def with_progress(iterable, desc: str, unit: str = 'line'):
	try:
		total = len(iterable)
	except TypeError:
		total = None

	return tqdm(iterable, desc=desc, total=total, unit=unit)


def configure_translation(model_id: str, model_location: str, batch_size: int = 4) -> None:
	global _model_id, _model_location, _batch_size
	_model_id = model_id
	_model_location = model_location
	_batch_size = max(1, int(batch_size or 1))


def normalize_language_label(value: str) -> str:
	return re.sub(r'[^a-z0-9]+', '-', (value or '').strip().lower()).strip('-')


def is_same_language(output_language: LanguageCode, output_language_raw: str, target_language: str) -> bool:
	target_language_code = LanguageCode.from_string(target_language)
	output_candidates = set()

	if output_language:
		output_candidates.add(normalize_language_label(output_language.to_iso_639_1()))
		output_candidates.add(normalize_language_label(output_language.to_iso_639_2_t()))
		output_candidates.add(normalize_language_label(output_language.to_iso_639_2_b()))
		output_candidates.add(normalize_language_label(output_language.to_name()))

	output_candidates.add(normalize_language_label(output_language_raw))
	output_candidates.discard('')

	if target_language_code:
		target_candidates = {
			normalize_language_label(target_language_code.to_iso_639_1()),
			normalize_language_label(target_language_code.to_iso_639_2_t()),
			normalize_language_label(target_language_code.to_iso_639_2_b()),
			normalize_language_label(target_language_code.to_name()),
		}
	else:
		target_candidates = {normalize_language_label(target_language)}

	target_candidates.discard('')
	return len(output_candidates.intersection(target_candidates)) > 0


def get_translation_model():
	global translation_model, translation_processor

	if translation_model is not None and translation_processor is not None:
		return translation_model, translation_processor

	with translation_model_lock:
		if translation_model is not None and translation_processor is not None:
			return translation_model, translation_processor

		try:
			from transformers import AutoModelForImageTextToText, AutoProcessor
		except ImportError:
			logging.error(
				'TRANSLATE_TO is set but transformers is not available. '
				'Install dependencies from requirements.txt to enable translation.'
			)
			return None, None

		load_kwargs = {
			'cache_dir': _model_location,
		}

		if torch.cuda.is_available():
			load_kwargs['device_map'] = 'auto'
			load_kwargs['dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

		translation_processor = AutoProcessor.from_pretrained(_model_id, cache_dir=_model_location)
		translation_model = AutoModelForImageTextToText.from_pretrained(_model_id, **load_kwargs)

		if not torch.cuda.is_available():
			translation_model.to('cpu')

		translation_model.eval()

		tokenizer = getattr(translation_processor, 'tokenizer', None)
		eos_token_id = getattr(tokenizer, 'eos_token_id', None)
		if eos_token_id is not None and getattr(translation_model.generation_config, 'pad_token_id', None) is None:
			translation_model.generation_config.pad_token_id = eos_token_id

		logging.info(f'Loaded TranslateGemma model: {_model_id}')
		return translation_model, translation_processor


def _build_translation_message(text: str, source_lang_code: str, target_lang_code: str) -> List[dict]:
	return [
		{
			'role': 'user',
			'content': [
				{
					'type': 'text',
					'source_lang_code': source_lang_code,
					'target_lang_code': target_lang_code,
					'text': text,
				}
			],
		}
	]


def _get_target_dtype(llm_model):
	if llm_model.device.type != 'cpu' and torch.cuda.is_bf16_supported():
		return torch.bfloat16
	return None


def _normalize_batch_size(batch_size: Optional[int]) -> int:
	if batch_size is None:
		return _batch_size
	return max(1, int(batch_size or 1))


def translate_texts_with_translategemma(
	texts: List[str],
	source_language: str,
	target_language: str,
	batch_size: Optional[int] = None,
	progress_desc: Optional[str] = None,
) -> List[str]:
	if not texts:
		return []

	llm_model, llm_processor = get_translation_model()
	cleaned_texts = [(text or '').strip() for text in texts]
	if llm_model is None or llm_processor is None:
		return cleaned_texts

	source_lang_code = to_translategemma_source_lang_code(source_language)
	target_lang_code = to_translategemma_target_lang_code(target_language)
	resolved_batch_size = _normalize_batch_size(batch_size)
	logging.info(
		'Translation started with fixed max_new_tokens=%s, TRANSLATE_BATCH_SIZE=%s '
		'(source=%s, target=%s, lines=%s)',
		_max_new_tokens,
		resolved_batch_size,
		source_lang_code,
		target_lang_code,
		len(texts),
	)

	translated_lines = [''] * len(cleaned_texts)
	non_empty_items = [(idx, text) for idx, text in enumerate(cleaned_texts) if text]
	if not non_empty_items:
		return translated_lines

	target_dtype = _get_target_dtype(llm_model)
	batch_offsets = list(range(0, len(non_empty_items), resolved_batch_size))
	iterable = with_progress(batch_offsets, progress_desc, unit='batch') if progress_desc else batch_offsets

	for start in iterable:
		batch_items = non_empty_items[start:start + resolved_batch_size]
		batch_messages = [
			_build_translation_message(text, source_lang_code, target_lang_code)
			for _, text in batch_items
		]

		with translation_model_lock:
			inputs = llm_processor.apply_chat_template(
				batch_messages,
				tokenize=True,
				add_generation_prompt=True,
				return_dict=True,
				return_tensors='pt',
				padding=True,
			)
			if target_dtype is not None:
				inputs = inputs.to(llm_model.device, dtype=target_dtype)
			else:
				inputs = inputs.to(llm_model.device)

			input_len = int(inputs['input_ids'].shape[1])

			tokenizer = getattr(llm_processor, 'tokenizer', None)
			eos_token_id = getattr(tokenizer, 'eos_token_id', None)
			pad_token_id = getattr(tokenizer, 'pad_token_id', None)
			if pad_token_id is None:
				pad_token_id = eos_token_id

			with torch.inference_mode():
				generate_kwargs = {
					'max_new_tokens': _max_new_tokens,
					'do_sample': False,
				}
				if pad_token_id is not None:
					generate_kwargs['pad_token_id'] = pad_token_id
				if eos_token_id is not None:
					generate_kwargs['eos_token_id'] = eos_token_id

				output_ids = llm_model.generate(
					**inputs,
					**generate_kwargs,
				)

		for row_idx, (original_idx, original_text) in enumerate(batch_items):
			generated_ids = output_ids[row_idx][input_len:]
			translated = llm_processor.decode(generated_ids, skip_special_tokens=True).strip().strip('"')
			translated_lines[original_idx] = translated if translated else original_text

	return translated_lines


def to_translategemma_source_lang_code(language: str) -> str:
	lang = LanguageCode.from_string(language)
	if lang:
		return lang.to_iso_639_1()

	normalized = (language or '').strip()
	if '-' in normalized:
		normalized = normalized.split('-', 1)[0]

	lang = LanguageCode.from_string(normalized)
	if lang:
		return lang.to_iso_639_1()

	return normalized.lower() or 'en'


def to_translategemma_target_lang_code(language: str) -> str:
	normalized = (language or '').strip()
	if '-' in normalized:
		return normalized

	lang = LanguageCode.from_string(normalized)
	if lang:
		return lang.to_iso_639_1()

	return normalized.lower() or 'en'


def translate_text_with_translategemma(text: str, source_language: str, target_language: str) -> str:
	translated_lines = translate_texts_with_translategemma(
		texts=[text],
		source_language=source_language,
		target_language=target_language,
		batch_size=1,
	)
	return translated_lines[0] if translated_lines else ''


def format_srt_timestamp(seconds: float) -> str:
	milliseconds = int(round(seconds * 1000))
	hours, remainder = divmod(milliseconds, 3600000)
	minutes, remainder = divmod(remainder, 60000)
	secs, ms = divmod(remainder, 1000)
	return f'{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}'


def name_bilingual_subtitle(
	file_path: str,
	source_language: LanguageCode,
	target_language: str,
	whisper_model: str,
	show_in_subname_subgen: bool,
	show_in_subname_model: bool,
) -> str:
	subgen_part = '.subgen' if show_in_subname_subgen else ''
	model_part = f'.{whisper_model}' if show_in_subname_model else ''
	source_part = source_language.to_iso_639_1() if source_language else 'src'

	target_language_code = LanguageCode.from_string(target_language)
	if target_language_code:
		target_part = target_language_code.to_iso_639_1()
	else:
		target_part = normalize_language_label(target_language) or 'target'

	return f'{os.path.splitext(file_path)[0]}{subgen_part}{model_part}.bilingual.{source_part}-{target_part}.srt'


def name_translated_subtitle(
	file_path: str,
	target_language: str,
	whisper_model: str,
	show_in_subname_subgen: bool,
	show_in_subname_model: bool,
) -> str:
	subgen_part = '.subgen' if show_in_subname_subgen else ''
	model_part = f'.{whisper_model}' if show_in_subname_model else ''

	target_language_code = LanguageCode.from_string(target_language)
	if target_language_code:
		target_part = target_language_code.to_iso_639_1()
	else:
		target_part = normalize_language_label(target_language) or 'target'

	return f'{os.path.splitext(file_path)[0]}{subgen_part}{model_part}.{target_part}.srt'


def write_bilingual_srt(result, translated_lines: List[str], output_path: str) -> None:
	with open(output_path, 'w', encoding='utf-8') as file:
		for idx, segment in enumerate(result.segments, start=1):
			translated_line = translated_lines[idx - 1] if idx - 1 < len(translated_lines) else ''
			original_line = (segment.text or '').strip()

			file.write(f'{idx}\n')
			file.write(f'{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}\n')
			if translated_line:
				file.write(f'{original_line}\n{translated_line}\n\n')
			else:
				file.write(f'{original_line}\n\n')


def write_translated_only_srt(result, translated_lines: List[str], output_path: str) -> None:
	with open(output_path, 'w', encoding='utf-8') as file:
		for idx, segment in enumerate(result.segments, start=1):
			translated_line = translated_lines[idx - 1] if idx - 1 < len(translated_lines) else ''
			file.write(f'{idx}\n')
			file.write(f'{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}\n')
			file.write(f'{translated_line}\n\n')


def create_bilingual_subtitle_if_needed(
	result,
	file_path: str,
	translate_to: str,
	whisper_model: str,
	show_in_subname_subgen: bool,
	show_in_subname_model: bool,
) -> Optional[str]:
	if not translate_to:
		return None

	output_language = LanguageCode.from_string(result.language)
	if is_same_language(output_language, result.language, translate_to):
		return None

	source_language = output_language.to_name() if output_language else (result.language or 'Unknown')
	translated_lines = translate_texts_with_translategemma(
		texts=[segment.text for segment in result.segments],
		source_language=source_language,
		target_language=translate_to,
		progress_desc=f'Translating subtitles to {translate_to}',
	)

	bilingual_subtitle_path = name_bilingual_subtitle(
		file_path=file_path,
		source_language=output_language,
		target_language=translate_to,
		whisper_model=whisper_model,
		show_in_subname_subgen=show_in_subname_subgen,
		show_in_subname_model=show_in_subname_model,
	)
	write_bilingual_srt(result, translated_lines, bilingual_subtitle_path)

	translated_only_path = name_translated_subtitle(
		file_path=file_path,
		target_language=translate_to,
		whisper_model=whisper_model,
		show_in_subname_subgen=show_in_subname_subgen,
		show_in_subname_model=show_in_subname_model,
	)
	write_translated_only_srt(result, translated_lines, translated_only_path)
	logging.info(f'Created translated-only subtitle: {translated_only_path}')

	return bilingual_subtitle_path


def parse_srt_cues(srt_text: str) -> List[dict]:
	lines = srt_text.splitlines()
	cues = []
	i = 0
	while i < len(lines):
		if not lines[i].strip():
			i += 1
			continue

		index_line = lines[i].strip()
		i += 1
		if i >= len(lines):
			break

		time_line = lines[i].strip()
		i += 1

		text_lines = []
		while i < len(lines) and lines[i].strip() != '':
			text_lines.append(lines[i].rstrip('\n'))
			i += 1

		cues.append(
			{
				'index': index_line,
				'time': time_line,
				'text_lines': text_lines,
			}
		)

		while i < len(lines) and lines[i].strip() == '':
			i += 1

	return cues


def write_merged_srt_from_cues(cues: List[dict], translated_lines: List[str], output_path: str) -> None:
	with open(output_path, 'w', encoding='utf-8') as file:
		for idx, cue in enumerate(cues):
			file.write(f"{cue['index']}\n")
			file.write(f"{cue['time']}\n")

			original_text = '\n'.join(cue['text_lines']).strip()
			translated_text = translated_lines[idx] if idx < len(translated_lines) else ''

			if original_text:
				file.write(f"{original_text}\n")
			if translated_text:
				file.write(f"{translated_text}\n")

			file.write('\n')


def write_translated_only_srt_from_cues(cues: List[dict], translated_lines: List[str], output_path: str) -> None:
	with open(output_path, 'w', encoding='utf-8') as file:
		for idx, cue in enumerate(cues):
			translated_text = translated_lines[idx] if idx < len(translated_lines) else ''
			file.write(f"{cue['index']}\n")
			file.write(f"{cue['time']}\n")
			file.write(f"{translated_text}\n\n")


def build_cli_output_path(input_srt_path: str, target_language: str) -> str:
	target_lang = LanguageCode.from_string(target_language)
	target_part = target_lang.to_iso_639_1() if target_lang else normalize_language_label(target_language)
	target_part = target_part or 'target'
	base, ext = os.path.splitext(input_srt_path)
	return f'{base}.bilingual.{target_part}{ext or ".srt"}'


def translate_srt_file_to_bilingual(
	input_srt_path: str,
	target_language: str,
	output_path: Optional[str] = None,
	source_language: str = '',
) -> str:
	if not os.path.exists(input_srt_path):
		raise FileNotFoundError(f'SRT file not found: {input_srt_path}')

	if not source_language or not source_language.strip():
		raise ValueError('source_language is required for SRT translation.')

	with open(input_srt_path, 'r', encoding='utf-8-sig') as file:
		srt_text = file.read()

	cues = parse_srt_cues(srt_text)
	if not cues:
		raise ValueError('No valid SRT cues found in input file.')

	translated_lines = translate_texts_with_translategemma(
		texts=['\n'.join(cue['text_lines']).strip() for cue in cues],
		source_language=source_language,
		target_language=target_language,
		progress_desc=f'Translating SRT to {target_language}',
	)

	final_output_path = output_path or build_cli_output_path(input_srt_path, target_language)
	write_merged_srt_from_cues(cues, translated_lines, final_output_path)

	base, ext = os.path.splitext(final_output_path)
	# Strip '.bilingual' suffix if present to derive the translated-only path
	if base.endswith('.bilingual'):
		base = base[: -len('.bilingual')]
	target_lang = LanguageCode.from_string(target_language)
	target_part = target_lang.to_iso_639_1() if target_lang else (normalize_language_label(target_language) or 'target')
	translated_only_path = f'{base}.{target_part}{ext or ".srt"}'
	write_translated_only_srt_from_cues(cues, translated_lines, translated_only_path)
	logging.info(f'Created translated-only subtitle: {translated_only_path}')

	return final_output_path

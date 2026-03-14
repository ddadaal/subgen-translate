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


def with_progress(iterable, desc: str):
	try:
		total = len(iterable)
	except TypeError:
		total = None

	return tqdm(iterable, desc=desc, total=total, unit='line')


def configure_translation(model_id: str, model_location: str, max_new_tokens: int) -> None:
	global _model_id, _model_location, _max_new_tokens
	_model_id = model_id
	_model_location = model_location
	_max_new_tokens = max_new_tokens


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

		tokenizer = getattr(translation_processor, 'tokenizer', None)
		eos_token_id = getattr(tokenizer, 'eos_token_id', None)
		if eos_token_id is not None and getattr(translation_model.generation_config, 'pad_token_id', None) is None:
			translation_model.generation_config.pad_token_id = eos_token_id

		logging.info(f'Loaded TranslateGemma model: {_model_id}')
		return translation_model, translation_processor


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
	clean_text = (text or '').strip()
	if not clean_text:
		return ''

	llm_model, llm_processor = get_translation_model()
	if llm_model is None or llm_processor is None:
		return clean_text

	source_lang_code = to_translategemma_source_lang_code(source_language)
	target_lang_code = to_translategemma_target_lang_code(target_language)

	messages = [
		{
			'role': 'user',
			'content': [
				{
					'type': 'text',
					'source_lang_code': source_lang_code,
					'target_lang_code': target_lang_code,
					'text': clean_text,
				}
			],
		}
	]

	target_dtype = torch.bfloat16 if llm_model.device.type != 'cpu' and torch.cuda.is_bf16_supported() else None

	with translation_model_lock:
		inputs = llm_processor.apply_chat_template(
			messages,
			tokenize=True,
			add_generation_prompt=True,
			return_dict=True,
			return_tensors='pt',
		)
		if target_dtype is not None:
			inputs = inputs.to(llm_model.device, dtype=target_dtype)
		else:
			inputs = inputs.to(llm_model.device)

		input_len = len(inputs['input_ids'][0])
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

	generated_ids = output_ids[0][input_len:]
	translated = llm_processor.decode(generated_ids, skip_special_tokens=True).strip().strip('"')
	return translated if translated else clean_text


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
	translated_lines = []
	for segment in with_progress(result.segments, f'Translating subtitles to {translate_to}'):
		translated_lines.append(
			translate_text_with_translategemma(segment.text, source_language, translate_to)
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

	translated_lines = []
	for cue in with_progress(cues, f'Translating SRT to {target_language}'):
		text = '\n'.join(cue['text_lines']).strip()
		translated_lines.append(
			translate_text_with_translategemma(text, source_language, target_language)
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

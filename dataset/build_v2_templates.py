#!/usr/bin/env python3
"""Build data/dataset_v2/templates_localized.json from the v1 file.

v2 changes (Utsav 2026-10-20): drop fr/de/ru; add pt/mt/ur/lo; add the
keyword_extraction and schema_constrained_output tasks for every language.
The pt/mt/ur/lo shells and all keyword_extraction shells were written in-session
(not human reviewed): mt and lo especially need a native-speaker check.

  .venv/bin/python dataset/build_v2_templates.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "dataset_v1" / "templates_localized.json"
V2 = ROOT / "data" / "dataset_v2" / "templates_localized.json"

LANGS = ["en", "es", "pt", "ja", "zh", "ar", "th", "hi", "id", "tr", "ur", "yo", "sw", "ne", "mt", "lo"]
NEW = {
    "pt": {
        "no_task": "{content}",
        "sentiment_analysis": "Analise o sentimento do texto a seguir e responda com 'positivo', 'negativo' ou 'neutro': '{content}'",
        "rephrase": "Reescreva o texto a seguir em um estilo clássico e arcaico do português: '{content}'",
        "summarize": "Resuma o texto a seguir em uma frase: '{content}'",
        "translate": "Traduza o texto a seguir para {translate_target_name}: '{content}'",
        "pos_tagging": "Identifique a classe gramatical de cada palavra no texto a seguir e retorne o resultado como pares palavra-etiqueta: '{content}'",
        "word_shuffling": "Embaralhe aleatoriamente a ordem das palavras no texto a seguir, preservando todas as palavras originais exatamente uma vez: '{content}'",
        "character_counting": "Conte o número total de caracteres no texto a seguir e responda apenas com o número: '{content}'",
    },
    "ur": {
        "no_task": "{content}",
        "sentiment_analysis": "درج ذیل متن کے جذبات کا تجزیہ کریں اور 'مثبت'، 'منفی' یا 'غیر جانبدار' میں جواب دیں: '{content}'",
        "rephrase": "درج ذیل متن کو کلاسیکی اور قدیم اردو اسلوب میں دوبارہ لکھیں: '{content}'",
        "summarize": "درج ذیل متن کا ایک جملے میں خلاصہ کریں: '{content}'",
        "translate": "درج ذیل متن کا {translate_target_name} میں ترجمہ کریں: '{content}'",
        "pos_tagging": "درج ذیل متن میں ہر لفظ کی اقسامِ کلام (part-of-speech) کی نشاندہی کریں اور نتیجہ لفظ–ٹیگ جوڑوں کی صورت میں دیں: '{content}'",
        "word_shuffling": "درج ذیل متن میں الفاظ کی ترتیب کو بے ترتیب طور پر بدلیں، مگر تمام اصل الفاظ ٹھیک ایک بار محفوظ رکھیں: '{content}'",
        "character_counting": "درج ذیل متن میں حروف کی کل تعداد گنیں اور صرف عدد کے ساتھ جواب دیں: '{content}'",
    },
    "mt": {
        "no_task": "{content}",
        "sentiment_analysis": "Analizza s-sentiment tat-test li ġej u wieġeb bi 'pożittiv', 'negattiv' jew 'newtrali': '{content}'",
        "rephrase": "Erġa' ikteb it-test li ġej fi stil Malti klassiku u arkajku: '{content}'",
        "summarize": "Agħmel sommarju tat-test li ġej f'sentenza waħda: '{content}'",
        "translate": "Ittraduċi t-test li ġej għall-{translate_target_name}: '{content}'",
        "pos_tagging": "Identifika l-parti tad-diskors ta' kull kelma fit-test li ġej u ġib ir-riżultat bħala pari kelma–tikketta: '{content}'",
        "word_shuffling": "Ħallat b'mod każwali l-ordni tal-kliem fit-test li ġej billi żżomm kull kelma oriġinali eżattament darba waħda: '{content}'",
        "character_counting": "Għodd in-numru totali ta' karattri fit-test li ġej u wieġeb bin-numru biss: '{content}'",
    },
    "lo": {
        "no_task": "{content}",
        "sentiment_analysis": "ວິເຄາະຄວາມຮູ້ສຶກຂອງຂໍ້ຄວາມຕໍ່ໄປນີ້ ແລ້ວຕອບວ່າ 'ບວກ', 'ລົບ' ຫຼື 'ເປັນກາງ': '{content}'",
        "rephrase": "ຂຽນຂໍ້ຄວາມຕໍ່ໄປນີ້ຄືນໃໝ່ໃນຮູບແບບພາສາລາວຄລາສສິກແບບບູຮານ: '{content}'",
        "summarize": "ສະຫຼຸບຂໍ້ຄວາມຕໍ່ໄປນີ້ໃນໜຶ່ງປະໂຫຍກ: '{content}'",
        "translate": "ແປຂໍ້ຄວາມຕໍ່ໄປນີ້ເປັນພາສາ{translate_target_name}: '{content}'",
        "pos_tagging": "ລະບຸຊະນິດຂອງຄໍາ (part-of-speech) ສໍາລັບແຕ່ລະຄໍາໃນຂໍ້ຄວາມຕໍ່ໄປນີ້ ແລະ ສົ່ງຜົນເປັນຄູ່ຄໍາ-ແທັກ: '{content}'",
        "word_shuffling": "ສຸ່ມສະລັບລໍາດັບຄໍາໃນຂໍ້ຄວາມຕໍ່ໄປນີ້ ໂດຍຮັກສາຄໍາເດີມທັງໝົດໄວ້ຢ່າງລະໜຶ່ງຄັ້ງ: '{content}'",
        "character_counting": "ນັບຈໍານວນຕົວອັກສອນທັງໝົດໃນຂໍ້ຄວາມຕໍ່ໄປນີ້ ແລະ ຕອບເປັນຕົວເລກເທົ່ານັ້ນ: '{content}'",
    },
}
KEYWORD = {
    "en": "Extract the three most important keywords from the following text and respond with only a comma-separated list: '{content}'",
    "es": "Extrae las tres palabras clave más importantes del siguiente texto y responde solo con una lista separada por comas: '{content}'",
    "pt": "Extraia as três palavras-chave mais importantes do texto a seguir e responda apenas com uma lista separada por vírgulas: '{content}'",
    "ja": "次のテキストから最も重要なキーワードを3つ抽出し、カンマ区切りのリストのみで答えてください: '{content}'",
    "zh": "从以下文本中提取三个最重要的关键词，并仅以逗号分隔的列表回答：'{content}'",
    "ar": "استخرج أهم ثلاث كلمات مفتاحية من النص التالي وأجب بقائمة مفصولة بفواصل فقط: '{content}'",
    "th": "ดึงคำสำคัญที่สำคัญที่สุดสามคำจากข้อความต่อไปนี้ และตอบเป็นรายการที่คั่นด้วยจุลภาคเท่านั้น: '{content}'",
    "hi": "निम्नलिखित पाठ से तीन सबसे महत्वपूर्ण कीवर्ड निकालें और केवल अल्पविराम से अलग की गई सूची के रूप में उत्तर दें: '{content}'",
    "id": "Ekstrak tiga kata kunci terpenting dari teks berikut dan jawab hanya dengan daftar yang dipisahkan koma: '{content}'",
    "tr": "Aşağıdaki metinden en önemli üç anahtar kelimeyi çıkarın ve yalnızca virgülle ayrılmış bir liste olarak yanıtlayın: '{content}'",
    "ur": "درج ذیل متن سے تین اہم ترین کلیدی الفاظ نکالیں اور صرف کوما سے الگ کی گئی فہرست کی صورت میں جواب دیں: '{content}'",
    "yo": "Yọ àwọn ọ̀rọ̀-àmì pàtàkì mẹ́ta jùlọ jáde láti inú ọ̀rọ̀ yìí, kí o sì dáhùn pẹ̀lú àtòjọ tí a ya sọ́tọ̀ pẹ̀lú kọ́mà nìkan: '{content}'",
    "sw": "Toa maneno matatu muhimu zaidi kutoka kwenye maandishi yafuatayo na ujibu kwa orodha iliyotenganishwa kwa koma pekee: '{content}'",
    "ne": "निम्नलिखित पाठबाट तीन सबैभन्दा महत्त्वपूर्ण कुञ्जी शब्द निकाल्नुहोस् र अल्पविरामले छुट्याइएको सूचीमा मात्र उत्तर दिनुहोस्: '{content}'",
    "mt": "Oħroġ it-tliet kliem ewlenin l-aktar importanti mit-test li ġej u wieġeb biss b'lista separata b'virgoli: '{content}'",
    "lo": "ດຶງຄໍາສໍາຄັນທີ່ສໍາຄັນທີ່ສຸດສາມຄໍາຈາກຂໍ້ຄວາມຕໍ່ໄປນີ້ ແລະ ຕອບເປັນລາຍການທີ່ແຍກດ້ວຍເຄື່ອງໝາຍຈຸດເທົ່ານັ້ນ: '{content}'",
}
SCHEMA = {
    "en": "Return the following text as a JSON object with a single key \"text\": '{content}'",
    "es": "Devuelve el siguiente texto como un objeto JSON con una única clave \"text\": '{content}'",
    "pt": "Retorne o texto a seguir como um objeto JSON com uma única chave \"text\": '{content}'",
    "ja": "次のテキストを、キーが \"text\" だけのJSONオブジェクトとして返してください: '{content}'",
    "zh": "请将以下文本作为只有一个键 \"text\" 的JSON对象返回：'{content}'",
    "ar": "أعد النص التالي ككائن JSON يحتوي على مفتاح واحد فقط هو \"text\": '{content}'",
    "th": "ส่งข้อความต่อไปนี้กลับมาเป็นอ็อบเจกต์ JSON ที่มีคีย์เดียวคือ \"text\": '{content}'",
    "hi": "निम्नलिखित पाठ को एक JSON ऑब्जेक्ट के रूप में लौटाएँ जिसमें केवल एक कुंजी \"text\" हो: '{content}'",
    "id": "Kembalikan teks berikut sebagai objek JSON dengan satu kunci saja, \"text\": '{content}'",
    "tr": "Aşağıdaki metni yalnızca tek bir \"text\" anahtarı olan bir JSON nesnesi olarak döndürün: '{content}'",
    "ur": "درج ذیل متن کو ایک JSON آبجیکٹ کے طور پر واپس کریں جس میں صرف ایک کی \"text\" ہو: '{content}'",
    "yo": "Dá ọ̀rọ̀ tó tẹ̀lé yìí padà gẹ́gẹ́ bí ohun JSON tí ó ní kọ́kọ́rọ́ kan ṣoṣo, \"text\": '{content}'",
    "sw": "Rudisha maandishi yafuatayo kama kitu cha JSON chenye ufunguo mmoja tu, \"text\": '{content}'",
    "ne": "निम्न पाठलाई एउटै कुञ्जी \"text\" भएको JSON अब्जेक्टका रूपमा फर्काउनुहोस्: '{content}'",
    "mt": "Irritorna t-test li ġej bħala oġġett JSON b'ċavetta waħda biss, \"text\": '{content}'",
    "lo": "ສົ່ງຄືນຂໍ້ຄວາມຕໍ່ໄປນີ້ເປັນອອບເຈັກ JSON ທີ່ມີຄີດຽວຄື \"text\": '{content}'",
}
# Only the (source -> translate target) pairs from the seeded round-robin
# (assign_translate_targets in build_dataset.py) whose target is a new language.
# Written in the source language, as it appears after the "translate" shell.
TARGET_NAMES = {
    "en": {"ur": "Urdu"},
    "pt": {"sw": "suaíli"},
    "zh": {"mt": "马耳他语"},
    "ar": {"lo": "اللاوية"},
    "id": {"pt": "Portugis"},
    "ur": {"zh": "چینی"},
    "mt": {"id": "Indoneżjan"},
    "lo": {"tr": "ເຕີກີ"},
}

# Per-language archaic target for the rephrase task (Kaustubh 2026-10-26: styles differ by
# language on purpose, a shared target would collapse rephrase into translate). Written in
# session, not native-reviewed; yo especially. Overrides v1 for the langs that had one.
REPHRASE = {
    "en": "Rewrite the following using Shakespearean English: '{content}'",
    "es": "Reescribe el siguiente texto en el español del Siglo de Oro, al estilo de Cervantes: '{content}'",
    "pt": "Reescreva o texto a seguir em português arcaico, ao estilo de Camões: '{content}'",
    "ja": "次のテキストを平安時代の古文（『源氏物語』のような文体）で書き直してください: '{content}'",
    "zh": "请用文言文（古典中文）风格改写以下文本：'{content}'",
    "ar": "أعد صياغة النص التالي بأسلوب المقامات العربية القديمة: '{content}'",
    "th": "เขียนข้อความต่อไปนี้ใหม่ด้วยสำนวนไทยโบราณแบบวรรณคดี เช่น ขุนช้างขุนแผน: '{content}'",
    "hi": "निम्नलिखित पाठ को संस्कृतनिष्ठ पुरानी साहित्यिक हिंदी शैली में पुनर्लेखित करें: '{content}'",
    "id": "Tulis ulang teks berikut dalam gaya bahasa Melayu klasik seperti dalam hikayat: '{content}'",
    "tr": "Aşağıdaki metni Osmanlı Türkçesi üslubuyla yeniden yazın: '{content}'",
    "ur": "درج ذیل متن کو قدیم کلاسیکی اردو نثر کے اسلوب میں، جیسے فسانۂ عجائب، دوبارہ لکھیں: '{content}'",
    "yo": "Tún ọ̀rọ̀ tó tẹ̀lé yìí kọ ní ọ̀nà Yorùbá àtijọ́: '{content}'",
    "sw": "Andika upya maandishi yafuatayo kwa mtindo wa Kiswahili cha kale cha tenzi za zamani, kama Utendi wa Tambuka: '{content}'",
    "ne": "निम्न पाठलाई पुरानो नेपाली साहित्यिक शैलीमा, भानुभक्त आचार्यको रामायणजस्तै, पुनर्लेखन गर्नुहोस्: '{content}'",
    "mt": "Erġa' ikteb it-test li ġej bi stil Malti qadim, bħal dak tal-Kantilena ta' Pietru Caxaro: '{content}'",
    "lo": "ຂຽນຂໍ້ຄວາມຕໍ່ໄປນີ້ຄືນໃໝ່ໃນພາສາລາວບູຮານແບບວັນນະຄະດີ ເຊັ່ນ ສິນໄຊ: '{content}'",
}


def main() -> None:
    raw = json.loads(V1.read_text())
    templates = {l: dict(raw["templates"][l]) for l in LANGS if l in raw["templates"]}
    templates.update({l: dict(t) for l, t in NEW.items()})
    for l in LANGS:
        templates[l]["rephrase"] = REPHRASE[l]
        templates[l]["keyword_extraction"] = KEYWORD[l]
        templates[l]["schema_constrained_output"] = SCHEMA[l]
    lang_names = {l: dict(raw["lang_names"].get(l, {})) for l in LANGS}
    for src, m in TARGET_NAMES.items():
        lang_names.setdefault(src, {}).update(m)
    tasks = list(raw["tasks"]) + ["keyword_extraction", "schema_constrained_output"]
    out = {
        "pipeline_version": "dataset-pipeline-v2",
        "template_sources": {
            "v1_kept": "data/dataset_v1/templates_localized.json",
            "written_in_session_unreviewed": ["pt", "mt", "ur", "lo", "rephrase (all langs, 2026-10-26)", "keyword_extraction (all langs)", "schema_constrained_output (all langs)"],
        },
        "tasks": tasks,
        "langs": LANGS,
        "templates": templates,
        "lang_names": lang_names,
    }
    V2.parent.mkdir(parents=True, exist_ok=True)
    V2.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    missing = [(l, t) for l in LANGS for t in tasks if t not in templates[l]]
    assert not missing, missing
    print("wrote", V2, "langs", len(LANGS), "tasks", len(tasks))


if __name__ == "__main__":
    main()

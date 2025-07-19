import os
import time
import pdfplumber
from flask import Flask, request, jsonify
from flask_cors import CORS
import string
from collections import Counter
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os
# إزالة استيراد llama_cpp وأي كود متعلق به

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# بيانات تدريبية مصغرة (يمكنك زيادتها لاحقاً)
TRAIN_DATA = [
    # مميزات
    ("يساعد على توفير الوقت والجهد", "feature"),
    ("يتيح للمستخدم سهولة الاستخدام", "feature"),
    ("يتميز المشروع بالمرونة", "feature"),
    ("يوفر أمان عالي للبيانات", "feature"),
    ("يساهم في تحسين تجربة العملاء", "feature"),
    ("The system is fast and reliable", "feature"),
    ("Provides added value to the user", "feature"),
    # مخاطر
    ("قد يواجه المشروع تحديات في التسويق", "risk"),
    ("من التحديات ارتفاع التكلفة", "risk"),
    ("هناك منافسة قوية في السوق", "risk"),
    ("قد يؤدي ضعف البنية التحتية إلى مشاكل", "risk"),
    ("من المخاطر صعوبة التنفيذ", "risk"),
    ("High cost is a potential risk", "risk"),
    ("Competition in the market is a challenge", "risk"),
    # توصيات
    ("ينصح بتطوير واجهة المستخدم", "rec"),
    ("يفضل زيادة التوعية بالمنتج", "rec"),
    ("يجب تحسين خدمة العملاء", "rec"),
    ("ننصح بإجراء اختبارات دورية", "rec"),
    ("It is recommended to improve security", "rec"),
    ("You should focus on marketing", "rec"),
    ("من الأفضل إضافة مزايا جديدة", "rec"),
]

MODEL_PATH = "text_classifier_model.joblib"

# تدريب أو تحميل الموديل
if os.path.exists(MODEL_PATH):
    text_clf = joblib.load(MODEL_PATH)
else:
    X_train = [t[0] for t in TRAIN_DATA]
    y_train = [t[1] for t in TRAIN_DATA]
    text_clf = make_pipeline(TfidfVectorizer(), MultinomialNB())
    text_clf.fit(X_train, y_train)
    joblib.dump(text_clf, MODEL_PATH)

def extract_text_from_pdf(pdf_file_path):
    import pdfplumber
    text = ""
    # المحاولة الأولى: pdfplumber
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print("pdfplumber error:", e)
    # إذا النص قليل أو غير مفهوم، جرب PyMuPDF
    if len(text.strip()) < 20 or not any('\u0600' <= c <= '\u06FF' for c in text):  # تحقق من وجود حروف عربية
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_file_path)
            text2 = ""
            for page in doc:
                text2 += page.get_text()
            if len(text2.strip()) > len(text.strip()):
                text = text2
        except Exception as e:
            print("PyMuPDF error:", e)
    # إذا مازال النص غير كافٍ، يمكن لاحقًا إضافة دعم OCR
    return text.strip()

def deep_pdf_analysis(text):
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 10]
    summary = ' '.join(sentences[:5])
    paragraphs = [p for p in text.split('\n') if len(p.strip()) > 10]
    lines = [l for l in text.split('\n') if l.strip()]
    num_sentences = len(sentences)
    num_paragraphs = len(paragraphs)
    num_lines = len(lines)
    return {
        'pdf_summary': summary,
        'num_sentences': num_sentences,
        'num_paragraphs': num_paragraphs,
        'num_lines': num_lines
    }

def extract_features_and_risks_from_description(description):
    """
    يحلل وصف المشروع ويستخرج المميزات والمخاطر المحتملة بناءً على تحليل لغوي بسيط.
    """
    features = []
    risks = []
    desc = description.lower()
    # مميزات شائعة
    if any(word in desc for word in ["سهل الاستخدام", "مرن", "قابل للتطوير", "فعال", "سريع", "آمن", "تكلفة منخفضة", "مبتكر", "جذاب"]):
        features.append({
            'title': 'سهولة الاستخدام',
            'description': 'الوصف يشير إلى أن المشروع سهل الاستخدام أو مرن أو مبتكر.'
        })
    if any(word in desc for word in ["يحل مشكلة", "يخدم شريحة واسعة", "يقلل التكاليف", "يوفر الوقت", "يقدم قيمة مضافة"]):
        features.append({
            'title': 'قيمة مضافة',
            'description': 'المشروع يقدم قيمة مضافة أو يحل مشكلة حقيقية.'
        })
    if any(word in desc for word in ["آمن", "حماية", "خصوصية"]):
        features.append({
            'title': 'الخصوصية والأمان',
            'description': 'الوصف يبرز جانب الأمان أو حماية البيانات.'
        })
    # مخاطر شائعة
    if any(word in desc for word in ["منافسة قوية", "تكلفة عالية", "صعوبة التنفيذ", "تعقيد", "غير واضح", "مخاطر تقنية", "غير مجرب"]):
        risks.append({
            'title': 'مخاطر التنفيذ أو المنافسة',
            'description': 'الوصف يشير إلى وجود تحديات في التنفيذ أو منافسة قوية.'
        })
    if len(description.split()) < 30:
        risks.append({
            'title': 'وصف مختصر جداً',
            'description': 'الوصف قصير جداً ولا يعطي صورة واضحة عن المشروع.'
        })
    if not features:
        features.append({'title': 'لا توجد مميزات واضحة', 'description': 'لم يتم ذكر مميزات محددة في الوصف.'})
    if not risks:
        risks.append({'title': 'لا توجد مخاطر واضحة', 'description': 'لم يتم ذكر مخاطر محددة في الوصف.'})
    return features, risks

def smart_local_analysis(text, target_audience=None, pdf_deep=None, description_text=None):
    if not text:
        return {
            'word_count': 0,
            'summary': 'لا يوجد نص لتحليله.',
            'audience_match': 0,
            'keywords': [],
            'score': 0,
            'recommendations': ['أضف وصفاً مفصلاً للمشروع.'],
            'risks': [{'title': 'نقص التفاصيل', 'description': 'نقص التفاصيل قد يضعف فرص النجاح.', 'suggestions': ['أضف وصفاً مفصلاً']}],
            'advantages': [],
            'pdf_deep': pdf_deep or {}
        }
    text_clean = text.translate(str.maketrans('', '', string.punctuation))
    words = [w for w in text_clean.split() if len(w) > 2]
    word_count = len(words)
    unique_words = len(set(words))
    keywords = [w for w, c in Counter(words).most_common(10)]
    length_score = min(40, word_count // 5)
    variety_score = min(30, int((unique_words / max(1, word_count)) * 30))
    audience_match = 0
    if target_audience:
        audience_keywords = [a.strip() for a in target_audience.split(',') if a.strip()]
        matches = sum(1 for a in audience_keywords if a in text)
        audience_match = int((matches / max(1, len(audience_keywords))) * 100)
    audience_score = min(20, audience_match // 5)
    score = length_score + variety_score + audience_score + min(10, len(keywords))
    recommendations = []
    if word_count < 50:
        recommendations.append('أضف تفاصيل أكثر لوصف المشروع.')
    if variety_score < 15:
        recommendations.append('استخدم مفردات متنوعة أكثر.')
    if audience_match < 50:
        recommendations.append('ركز على ذكر الفئة المستهدفة بوضوح في الوصف.')
    if pdf_deep:
        if pdf_deep['num_paragraphs'] < 3:
            recommendations.append('ملف PDF يحتاج إلى مزيد من الفقرات والتقسيم.')
        if pdf_deep['num_sentences'] < 10:
            recommendations.append('ملف PDF يحتاج إلى مزيد من الشرح والتفصيل.')
        if len(keywords) > 0:
            recommendations.append(f"ركز على الكلمات المفتاحية التالية في تطوير مشروعك: {', '.join(keywords[:5])}")
    if not recommendations:
        recommendations.append('الوصف جيد، استمر في تطوير الفكرة.')
    risks = []
    if word_count < 30:
        risks.append({'title': 'الوصف قصير جداً', 'description': 'الوصف قصير جداً، قد لا يقنع المستثمرين.', 'suggestions': ['أضف تفاصيل أكثر']})
    if audience_match < 30:
        risks.append({'title': 'عدم وضوح الفئة المستهدفة', 'description': 'عدم وضوح الفئة المستهدفة قد يضعف فرص النجاح.', 'suggestions': ['حدد الفئة المستهدفة بوضوح']})
    if pdf_deep:
        if pdf_deep['num_paragraphs'] < 2:
            risks.append({'title': 'قلة الفقرات', 'description': 'ملف PDF يحتوي على فقرات قليلة جداً.', 'suggestions': ['قسّم النص إلى فقرات واضحة']})
        if pdf_deep['num_sentences'] < 5:
            risks.append({'title': 'قلة الشرح', 'description': 'ملف PDF يحتاج إلى مزيد من الشرح.', 'suggestions': ['أضف جمل توضيحية أكثر']})
    advantages = []
    if word_count > 100:
        advantages.append({'title': 'وصف مفصل', 'description': 'وصف المشروع مفصل وجيد.'})
    if variety_score > 20:
        advantages.append({'title': 'تنوع المفردات', 'description': 'هناك تنوع جيد في المفردات.'})
    if audience_match > 70:
        advantages.append({'title': 'وضوح الفئة المستهدفة', 'description': 'الفئة المستهدفة واضحة ومذكورة بوضوح.'})
    if pdf_deep:
        if pdf_deep['num_paragraphs'] > 5:
            advantages.append({'title': 'تنظيم ممتاز', 'description': 'ملف PDF منظم ويحتوي على عدة فقرات.'})
        if pdf_deep['num_sentences'] > 20:
            advantages.append({'title': 'شرح وافي', 'description': 'ملف PDF يحتوي على شرح وافي ومفصل.'})
    # تحليل المميزات والمخاطر من وصف المشروع فقط
    features_from_desc, risks_from_desc = ([], [])
    if description_text:
        features_from_desc, risks_from_desc = extract_features_and_risks_from_description(description_text)
    summary = smart_summary(text, keywords)
    return {
        'word_count': word_count,
        'summary': summary,
        'audience_match': audience_match,
        'keywords': keywords,
        'score': score,
        'recommendations': recommendations,
        'risks': risks + risks_from_desc,
        'advantages': advantages + features_from_desc,
        'pdf_deep': pdf_deep or {}
    }

def smart_summary(text, keywords):
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 10]
    ranked = sorted(sentences, key=lambda s: sum(1 for k in keywords if k in s), reverse=True)
    summary = ' '.join(ranked[:5]) if ranked else ' '.join(sentences[:5])
    return summary

def translate_to_arabic(text):
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        translated_sentences = []
        for sentence in sentences:
            params = {
                "client": "gtx",
                "sl": "auto",
                "tl": "ar",
                "dt": "t",
                "q": sentence
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                result = response.json()
                translated = ''.join([t[0] for t in result[0]])
                translated_sentences.append(translated)
            else:
                translated_sentences.append(sentence)
        return '. '.join(translated_sentences)
    except Exception as e:
        return f"خطأ في الترجمة: {str(e)}"

def analyze_text_block(text, text_clf):
    import re
    sentences = re.split(r'[.!؟\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    features, risks, recommendations = [], [], []
    for s in sentences:
        label = text_clf.predict([s])[0]
        if label == "feature":
            features.append(s)
        elif label == "risk":
            risks.append(s)
        elif label == "rec":
            recommendations.append(s)
    summary = ' '.join(sentences[:3]) if len(sentences) >= 3 else ' '.join(sentences)
    # منطق النسبة الواقعي
    score = 60
    score += min(25, len(features)*5)           # كل ميزة تزيد النسبة
    score += min(15, len(recommendations)*5)    # كل توصية تزيد النسبة
    score -= min(30, len(risks)*7)              # كل مخاطرة تقلل النسبة
    score = max(10, min(100, score))            # النسبة بين 10 و 100 فقط
    # تنسيق المميزات
    if features:
        features_text = "\n".join([f"• {f}" for f in features])
        features_text = f"المميزات:\n{features_text}"
    else:
        features_text = "المميزات:\nلم يتم العثور على مميزات واضحة في النص."
    # تنسيق المخاطر
    if risks:
        risks_text = "\n".join([f"⚠️ {r}" for r in risks])
        risks_text = f"المخاطر:\n{risks_text}"
    else:
        risks_text = "المخاطر:\nلم يتم العثور على مخاطر أو تحديات واضحة في النص."
    # تنسيق التوصيات
    if recommendations:
        recs_text = "\n".join([f"- {rec}" for rec in recommendations])
        recs_text = f"التوصيات:\n{recs_text}"
    else:
        recs_text = "التوصيات:\nلم يتم العثور على توصيات مباشرة."
    return {
        'summary': summary,
        'advantages': features,  # قائمة كما كان
        'risks': risks,          # قائمة كما كان
        'advantages_formatted': features_text,  # نص منسق
        'risks_formatted': risks_text,          # نص منسق
        'recommendations': recs_text,
        'score': score
    }

def smart_ai_like_analysis(description, pdf_text=None, target_audience=None):
    """
    تحليل منفصل للوصف النصي وPDF، مع إرجاع نتائج كل منهما بشكل منفصل.
    """
    desc_result = analyze_text_block(description or '', text_clf) if description else None
    pdf_result = analyze_text_block(pdf_text or '', text_clf) if pdf_text else None
    return {
        'description_analysis': desc_result,
        'pdf_analysis': pdf_result
    }

@app.route('/analyze', methods=['POST'])
def analyze_project():
    try:
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            project_name = request.form.get('projectTitle', '').strip()
            url = request.form.get('url', '').strip()
            description = request.form.get('description', '').strip()
            target_audience = request.form.get('target_audience', '').strip()
            pdf_files = request.files.getlist('pdf')
            if not description and not pdf_files:
                return jsonify({'success': False, 'error': 'يجب إدخال وصف للمشروع أو رفع ملف PDF.'}), 400
            pdf_analyses = []
            for pdf_file in pdf_files:
                temp_pdf_path = os.path.join(UPLOAD_FOLDER, f"temp_{int(time.time())}_{pdf_file.filename}")
                pdf_file.save(temp_pdf_path)
                pdf_text = extract_text_from_pdf(temp_pdf_path)
                pdf_deep = deep_pdf_analysis(pdf_text)
                pdf_result = smart_ai_like_analysis('', pdf_text, target_audience)['pdf_analysis']
                pdf_analyses.append({
                    'filename': pdf_file.filename,
                    'analysis': pdf_result,
                    'deep': pdf_deep
                })
                os.remove(temp_pdf_path)
        else:
            data = request.get_json()
            project_name = data.get('projectTitle', '').strip()
            url = data.get('url', '').strip()
            description = data.get('description', '').strip()
            target_audience = data.get('target_audience', '').strip()
            pdf_analyses = []
            if not description:
                return jsonify({'success': False, 'error': 'يجب إدخال وصف للمشروع أو رفع ملف PDF.'}), 400

        # حلل الوصف النصي دائماً
        ai_result = smart_ai_like_analysis(description, '', target_audience)

        result = {
            'project_name': project_name,
            'url': url,
            'description_analysis': ai_result['description_analysis'],
            'pdf_analyses': pdf_analyses,
            'ai_used': True
        }
        return jsonify(result)
    except Exception as e:
        print("خطأ:", e)
        return jsonify({'success': False, 'error': f'خطأ في الخادم: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000, host='0.0.0.0')

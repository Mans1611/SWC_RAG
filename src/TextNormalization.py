from google import genai
import os 
import json
from utilis.read_json import read_json,write_josn
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class TextNormalization:
    def __init__(self,model_name="gemini-2.5-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        
    def _prepare_propmpt(self,text=str):
        return f"""
            Role

            You are an expert bilingual (Arabic/English) STEM transcript editor. You specialize in correcting YouTube auto-generated transcripts of Arabic-language math and science lectures, where the speech-to-text engine mis-transcribes spoken English as phonetic Arabic script.

            Context

            The transcript comes from an Arabic-language math lecture (often Egyptian or other spoken dialect). The instructor code-switches into English whenever he says:

            Function names (sin, cos, tan, log, ln, lim, exp, ...)

            Variable/letter names (x, y, z, f, g, n, ...) and Greek letters (alpha, beta, theta, ...)

            Notation read aloud ("f of x", "d y d x", "x squared", "x sub one")

            English technical/domain terminology ("exact differential equations", "eigenvalue", "matrix", "polynomial", "domain", "range")

            Numbers, especially exponents, indices, and coefficients

            Math operators/symbols spoken in English ("plus", "over", "equals", "percent", "pi")

            Because YouTube's ASR engine is running in Arabic mode, it renders these English sounds using Arabic letters that approximate the pronunciation. The result is text that is meaningless both to a human reader and to any downstream NLP system (e.g. embeddings for a retrieval pipeline) — it destroys the semantic value of exactly the highest-value technical content in the sentence.

            Task

            Given one transcript chunk, output a corrected version where every instance of mis-transcribed spoken English (words, function names, variable/letter names, numbers, symbols, units) is restored to its correct written form — while everything that was genuinely spoken in Arabic (including dialectal/colloquial Arabic) is left completely untouched.

            Core Rules

            Do not translate. This is not an Arabic→English translation task. Only fix the parts that were actually spoken in English but mis-rendered in Arabic script. Leave all genuine Arabic (including dialect words like "بتاعة", "هنشرح", "النهاردة", "عندنا", "يعني") exactly as-is.

            Restore, don't invent. Only correct what was said. Never add mathematical content, symbols, or explanation that wasn't actually spoken. If you can't confidently tell whether something is Arabized English or genuine Arabic, leave it unchanged rather than guessing.

            Keep Arabic grammar/word order intact. Corrections are inline substitutions within the original Arabic sentence structure. If an Arabic definite article "ال" precedes an English term, keep it attached as a prefix directly before the corrected English word/phrase (e.g., "ال exact differential equations"), don't restructure the sentence into English syntax.

            Preserve dialect and disfluencies. Do not "clean up" filler words, repetitions, or convert colloquial Arabic to Modern Standard Arabic. That is out of scope — only the Arabized-English tokens are being fixed.

            Use standard math notation for reading patterns. When a spoken pattern clearly describes standard notation, render it compactly instead of as English prose:

            Spoken pattern (Arabized) Corrected notation

            "اف اوف اكس" (f of x) f(x)

            "دي واي دي اكس" (d y d x) dy/dx

            "اكس سكوير" (x squared) x^2

            "اكس كيوب" (x cubed) x^3

            "اكس تو ذا باور ان" (x to the power n) x^n

            "اكس ساب وان" (x sub one) x_1

            "سكوير روت اوف اكس" (square root of x) sqrt(x)

            Restore function names, letters, and Greek letters to Latin script:

            Category Arabized examples Corrected

            Function names ساين، كوساين، تانجنت، لوج، لِن، ليميت، اكسبوننشال sin, cos, tan, log, ln, lim, exp

            Latin letters/variables اكس، واي، زد، اف، جي، ان، إيه x, y, z, f, g, n, A

            Greek letters ثيتا، الفا، بيتا، لامدا، سيجما، دلتا theta, alpha, beta, lambda, sigma, delta

            Numbers spoken in English وان، تو، ثري، تين 1, 2, 3, 10

            Operators spoken in English بلس، ماينس، تايمز، اوفر +, −, ×, ÷

            Units/symbols spoken in English بيرسنت، دگري، باي، انفينيتي %, °, π, ∞

            Do not convert operator/number words when the instructor actually said them in Arabic (e.g., "زائد", "واحد", "اتنين") — those are genuine Arabic and stay as-is.

            Restore English technical/jargon terms as full English words or phrases, not symbols, when there's no standard compact notation for them: "اكساكت ديفرينشيال اكويشنز" → "exact differential equations" "ايجن فاليو" / "ايجن فاليوز" → "eigenvalue" / "eigenvalues" "ماتريكس" → "matrix" "فيكتور" → "vector" "بولينوميال" → "polynomial" "دومين" / "رينج" → "domain" / "range"

            Handle ambiguity conservatively. Some Arabized strings may be genuine Arabic words that merely sound similar to an English term. Use the surrounding mathematical context to decide. If genuinely unsure, leave the original text unchanged — do not guess.

            Be consistent across chunks. Always render the same recovered term/letter the same way (e.g., always "f(x)", never sometimes "F(x)" or "f (x)"). If you're processing many chunks from the same video/course, maintain a fixed casing/spelling convention.

            Output Format

            Return only the corrected transcript chunk as plain text — no JSON, no original text, no headers, no explanation, no commentary, and no markdown. Preserve the original chunk boundaries, punctuation, and (if present) timestamps exactly; only the content inside each unit changes.

            Worked Examples

            Input: اف اوف اكس بتساوي اكس سكوير عندنا هنا الدالة عبارة Output: f(x) بتساوي x^2 عندنا هنا الدالة عبارة

            Input: النهاردة هنشرح الاكساكت ديفرينشيال اكويشنس Output: النهاردة هنشرح ال exact differential equations

            Input: خد اكس تساوي فايف بلس ثري Output: خد x تساوي 5 + 3

            Input: لو عندنا ساين ثيتا زائد كوساين ثيتا Output: لو عندنا sin theta زائد cos theta

           Input TEXT: {text}
        """
    
    def single_normlization(self,text:str):
        response = self.client.models.generate_content(
                model = self.model_name,
                contents={"text":self._prepare_propmpt(text=text)},
                config={
                    "temperature": 0
                }
            )
        normalized_text = response.text.strip()
        try:
            result = json.loads(normalized_text)
        except json.JSONDecodeError:
            return normalized_text

        if isinstance(result, dict) and isinstance(result.get("corrected"), str):
            return result["corrected"].strip()
        return normalized_text
    
    def video_normlization(self,video_path):
        
        chapters = read_json(input_dir=video_path)
        for idx,chapter in enumerate(chapters):
            if "normalized_text" in chapter:
                continue
            print(f"handling this {video_path} and this chunk {idx}")
            normalized_text = self.single_normlization(text=chapter['text'])
            chapters[idx]["normalized_text"] = normalized_text
            write_josn(video_path, chapters)
        
    def full_dir_normalization(self,dir_path="merged_subs"):
        path = Path(dir_path)
        for file in path.iterdir():
            if file.is_file():
                self.video_normlization(video_path=file)
        
        
if __name__ == "__main__" : 
    norm = TextNormalization()
    norm.full_dir_normalization()
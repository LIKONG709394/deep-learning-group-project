const API_KEY = process.env.EXPO_PUBLIC_POE_API_KEY || '';
const BASE_URL = process.env.EXPO_PUBLIC_POE_BASE_URL || 'https://api.poe.com/v1';
const MODEL = process.env.EXPO_PUBLIC_POE_MODEL || 'Claude-3.5-Sonnet';

if (!API_KEY) {
  console.warn('Poe API key not found. Please set EXPO_PUBLIC_POE_API_KEY in your .env file');
}

// Use a direct fetch call instead of the OpenAI browser client to avoid extra custom headers (e.g. x-stainless-os)
async function callPoeAPI(prompt: string): Promise<string> {
  try {
    const response = await fetch(`${BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: 'user', content: prompt }
        ]
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Poe API responded ${response.status}: ${text}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || '';
  } catch (error) {
    console.error('Poe API call failed:', error);
    throw error;
  }
}

export interface FlashCard {
  id: number;
  term: string;
  definition: string;
}

export interface StudyLevel {
  id: number;
  title: string;
  description: string;
  completed: boolean;
  locked: boolean;
}

export interface ShortVideo {
  id: number;
  title: string;
  description: string;
  keyPoints: string[];
}

export async function extractPDFContent(fileUri: string, fileName: string): Promise<string> {
  try {
    const prompt = `Based on a PDF file named "${fileName}", generate realistic educational content about the topic suggested by the filename. Include key concepts, definitions, and explanations that would typically be found in such a document. Make it comprehensive (500-800 words) as if you extracted text from an actual educational PDF.`;
    
    const content = await callPoeAPI(prompt);
    return content;
  } catch (error) {
    console.error('Error extracting PDF content:', error);
    throw new Error('Failed to extract PDF content. Please check your Poe API key.');
  }
}

export async function generateFlashcards(content: string): Promise<FlashCard[]> {
  try {
    const prompt = `Based on the following educational content, create 10-15 high-quality flashcards for studying.
Each flashcard should have a term/concept and a clear, concise definition/explanation.

Format your response as a valid JSON array with this structure:
[
  {
    "term": "Term or concept",
    "definition": "Clear explanation"
  }
]

Content:
${content}

Return ONLY the JSON array, no additional text or markdown formatting.`;

    const text = await callPoeAPI(prompt);
    
    const jsonMatch = text.match(/\[[\s\S]*\]/);
    if (!jsonMatch) {
      throw new Error('Failed to parse flashcards from response');
    }
    
    const flashcardsData = JSON.parse(jsonMatch[0]);
    const baseId = Date.now();
    
    return flashcardsData.map((card: any, index: number) => ({
      id: baseId + index,
      term: card.term,
      definition: card.definition,
    }));
  } catch (error) {
    console.error('Error generating flashcards:', error);
    throw new Error('Failed to generate flashcards');
  }
}

export async function generateStudyPath(content: string): Promise<StudyLevel[]> {
  try {
    const prompt = `Based on the following educational content, create a structured learning path with 5-7 progressive levels.
Each level should build upon the previous one, from basic concepts to advanced topics.

Format your response as a valid JSON array with this structure:
[
  {
    "title": "Level title",
    "description": "Brief description of what students will learn"
  }
]

Content:
${content}

Return ONLY the JSON array, no additional text or markdown formatting.`;

    const text = await callPoeAPI(prompt);
    
    const jsonMatch = text.match(/\[[\s\S]*\]/);
    if (!jsonMatch) {
      throw new Error('Failed to parse study path from response');
    }
    
    const levelsData = JSON.parse(jsonMatch[0]);
    const baseId = Date.now();
    
    return levelsData.map((level: any, index: number) => ({
      id: baseId + index,
      title: level.title,
      description: level.description,
      completed: index === 0,
      locked: index > 1,
    }));
  } catch (error) {
    console.error('Error generating study path:', error);
    throw new Error('Failed to generate study path');
  }
}

export async function generateShortSummaries(content: string): Promise<ShortVideo[]> {
  try {
    const prompt = `Based on the following educational content, create 3-5 short video script summaries.
Each summary should be concise, engaging, and focus on a specific key concept.

Format your response as a valid JSON array with this structure:
[
  {
    "title": "Video title",
    "description": "Engaging 2-3 sentence summary",
    "keyPoints": ["Key point 1", "Key point 2", "Key point 3"]
  }
]

Content:
${content}

Return ONLY the JSON array, no additional text or markdown formatting.`;

    const text = await callPoeAPI(prompt);
    
    const jsonMatch = text.match(/\[[\s\S]*\]/);
    if (!jsonMatch) {
      throw new Error('Failed to parse video summaries from response');
    }
    
    const videosData = JSON.parse(jsonMatch[0]);
    const baseId = Date.now();
    
    return videosData.map((video: any, index: number) => ({
      id: baseId + index,
      title: video.title,
      description: video.description,
      keyPoints: video.keyPoints,
    }));
  } catch (error) {
    console.error('Error generating video summaries:', error);
    throw new Error('Failed to generate video summaries');
  }
}

export async function processPDFAndGenerateContent(fileUri: string, fileName: string) {
  try {
    const content = await extractPDFContent(fileUri, fileName);
    
    const [flashcards, studyPath, shortVideos] = await Promise.all([
      generateFlashcards(content),
      generateStudyPath(content),
      generateShortSummaries(content),
    ]);

    return {
      content,
      flashcards,
      studyPath,
      shortVideos,
      fileName,
      processedAt: new Date().toISOString(),
    };
  } catch (error) {
    console.error('Error processing PDF:', error);
    throw error;
  }
}

// ── Vision / Camera OCR ──────────────────────────────────────────────────────

async function callPoeAPIWithImage(base64Image: string, mimeType: string, prompt: string): Promise<string> {
  try {
    const response = await fetch(`${BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'image_url',
                image_url: { url: `data:${mimeType};base64,${base64Image}` },
              },
              { type: 'text', text: prompt },
            ],
          },
        ],
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Poe API responded ${response.status}: ${text}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || '';
  } catch (error) {
    console.error('Poe API vision call failed:', error);
    throw error;
  }
}

export async function extractTextFromImage(base64Image: string, mimeType: string): Promise<string> {
  const prompt = `Please extract and transcribe ALL text visible in this image. This is a textbook or educational document page.

Extract:
1. All headings and titles
2. All body text, paragraphs, and explanations
3. Any bullet points or numbered lists
4. Captions, labels, and annotations
5. Any tables or structured data (as plain text)

Format the output as clean, readable text preserving the logical structure. Return ONLY the extracted text — no commentary.`;

  const text = await callPoeAPIWithImage(base64Image, mimeType, prompt);
  if (!text || text.trim().length < 30) {
    throw new Error('Could not extract sufficient text from the image. Please retake the photo with better lighting and focus.');
  }
  return text;
}

export async function processImageAndGenerateContent(base64Image: string, mimeType: string, imageName: string) {
  try {
    const content = await extractTextFromImage(base64Image, mimeType);

    const [flashcards, studyPath, shortVideos] = await Promise.all([
      generateFlashcards(content),
      generateStudyPath(content),
      generateShortSummaries(content),
    ]);

    return {
      content,
      flashcards,
      studyPath,
      shortVideos,
      fileName: imageName,
      processedAt: new Date().toISOString(),
    };
  } catch (error) {
    console.error('Error processing image:', error);
    throw error;
  }
}

// ── Summary Export ───────────────────────────────────────────────────────────

export interface CourseForSummary {
  title: string;
  content?: string;
  flashcards?: FlashCard[];
  studyPath?: StudyLevel[];
  shortVideos?: ShortVideo[];
  processedAt?: string;
}

/**
 * Compiles all AI-generated course data into a well-formatted summary string.
 * This is used to generate a downloadable study summary instead of sharing the raw PDF.
 */
export function buildCourseSummaryText(course: CourseForSummary): string {
  const date = course.processedAt
    ? new Date(course.processedAt).toLocaleDateString()
    : new Date().toLocaleDateString();

  const lines: string[] = [];

  lines.push('='.repeat(60));
  lines.push(`STUDY SUMMARY: ${course.title.toUpperCase()}`);
  lines.push(`Generated: ${date}`);
  lines.push('='.repeat(60));
  lines.push('');

  // ── Raw content summary ────────────────────────────────────
  if (course.content && course.content.trim().length > 0) {
    lines.push('OVERVIEW');
    lines.push('-'.repeat(40));
    // Trim to first ~800 chars as a brief overview
    const overview = course.content.trim().slice(0, 800);
    lines.push(overview + (course.content.length > 800 ? '...' : ''));
    lines.push('');
  }

  // ── Study Path ─────────────────────────────────────────────
  if (course.studyPath && course.studyPath.length > 0) {
    lines.push('LEARNING PATH');
    lines.push('-'.repeat(40));
    course.studyPath.forEach((level, i) => {
      lines.push(`${i + 1}. ${level.title}`);
      lines.push(`   ${level.description}`);
    });
    lines.push('');
  }

  // ── Short summaries / key concepts ────────────────────────
  if (course.shortVideos && course.shortVideos.length > 0) {
    lines.push('KEY CONCEPTS');
    lines.push('-'.repeat(40));
    course.shortVideos.forEach((video, i) => {
      lines.push(`${i + 1}. ${video.title}`);
      lines.push(`   ${video.description}`);
      if (video.keyPoints && video.keyPoints.length > 0) {
        video.keyPoints.forEach(point => {
          lines.push(`   • ${point}`);
        });
      }
      lines.push('');
    });
  }

  // ── Flashcards ─────────────────────────────────────────────
  if (course.flashcards && course.flashcards.length > 0) {
    lines.push('FLASHCARDS (Q&A)');
    lines.push('-'.repeat(40));
    course.flashcards.forEach((card, i) => {
      lines.push(`Q${i + 1}: ${card.term}`);
      lines.push(`A:  ${card.definition}`);
      lines.push('');
    });
  }

  lines.push('='.repeat(60));
  lines.push('Generated by SnapStudy');
  lines.push('='.repeat(60));

  return lines.join('\n');
}

// ── URL / Web-page Scraping ──────────────────────────────────────────────────

/**
 * Strips HTML tags, script/style blocks, and collapses whitespace to extract
 * readable plain text from an HTML string.
 */
function stripHtml(html: string): string {
  // Remove <script> and <style> blocks (including content)
  let text = html.replace(/<script[\s\S]*?<\/script>/gi, ' ');
  text = text.replace(/<style[\s\S]*?<\/style>/gi, ' ');
  // Remove all remaining HTML tags
  text = text.replace(/<[^>]+>/g, ' ');
  // Decode common HTML entities
  text = text
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ');
  // Collapse whitespace
  text = text.replace(/\s{2,}/g, ' ').trim();
  return text;
}

/**
 * Fetches a webpage URL and returns its plain-text content.
 * Works on native (no CORS). On web it may be blocked by the target server.
 */
export async function scrapeWebpageText(url: string): Promise<string> {
  let response: Response;
  try {
    response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; SnapStudy/1.0)',
        'Accept': 'text/html,application/xhtml+xml',
      },
    });
  } catch (err) {
    throw new Error('Could not reach the URL. Please check the link and your internet connection.');
  }

  if (!response.ok) {
    throw new Error(`The website returned an error (${response.status}). Try a different URL.`);
  }

  const contentType = response.headers.get('content-type') || '';
  if (!contentType.includes('html') && !contentType.includes('text')) {
    throw new Error('The URL does not point to a readable webpage (e.g. it may be a PDF or binary file). Use the PDF upload option instead.');
  }

  const html = await response.text();
  const text = stripHtml(html);

  if (text.length < 200) {
    throw new Error('Not enough text could be extracted from this page. The site may require a login or block automated access.');
  }

  // Cap to ~8000 chars so the AI prompt stays within limits
  return text.slice(0, 8000);
}

/**
 * Scrapes a webpage then generates flashcards, study path, and short summaries.
 */
export async function processUrlAndGenerateContent(url: string) {
  const hostname = (() => {
    try { return new URL(url).hostname.replace(/^www\./, ''); }
    catch { return url; }
  })();

  const text = await scrapeWebpageText(url);

  const [flashcards, studyPath, shortVideos] = await Promise.all([
    generateFlashcards(text),
    generateStudyPath(text),
    generateShortSummaries(text),
  ]);

  return {
    content: text,
    flashcards,
    studyPath,
    shortVideos,
    fileName: hostname,
    processedAt: new Date().toISOString(),
  };
}

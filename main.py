import torch
from transformers import GPTNeoForCausalLM, AutoTokenizer

import textstat
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from nltk.corpus import stopwords, wordnet

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load GPT-Neo and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

def fixSentenceSpacing(text):
    return re.sub(r'([.!?])(?=\S)', r'\1 ', text)

def getEmotionScore(text):
    emotionKeywords = {
        'Happy': ['happy', 'joy', 'pleased', 'delighted', 'excited'],
        'Angry': ['angry', 'furious', 'rage', 'mad'],
        'Surprise': ['surprised', 'astonished', 'shocked'],
        'Sad': ['sad', 'depressed', 'unhappy', 'down'],
        'Fear': ['afraid', 'scared', 'fear', 'terrified']
    }
    wordList = word_tokenize(text.lower())
    emotionCounts = {emotion: 0 for emotion in emotionKeywords}
    for word in wordList:
        for emotion, keywords in emotionKeywords.items():
            if word in keywords:
                emotionCounts[emotion] += 1
    total = sum(emotionCounts.values())
    return total / len(wordList) if wordList else 0

def calculatePerplexity(sentence):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    return torch.exp(loss).item()

def lexicalRichness(sentence):
    words = word_tokenize(sentence.lower())
    if len(words) == 0:
        return 0
    uniqueWords = set(words)
    return len(uniqueWords) / len(words)

def grammarIssues(sentence):
    issues = 0
    if '  ' in sentence:
        issues += 1
    words = word_tokenize(sentence.lower())
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            issues += 1
    return issues

def analyzeText(text):
    text = fixSentenceSpacing(text)
    sentences = sent_tokenize(text)
    results = []
    aiCount = 0
    humanCount = 0
    totalPerplexity = 0
    totalEmotion = 0
    totalReadability = 0
    totalRichness = 0
    totalSentLength = 0
    totalGrammarIssues = 0

    for sentence in sentences:
        perplexity = calculatePerplexity(sentence)
        emotionStrength = getEmotionScore(sentence)
        readability = textstat.flesch_reading_ease(sentence)
        richness = lexicalRichness(sentence)
        sentLength = len(word_tokenize(sentence))
        grammarIssueCount = grammarIssues(sentence)

        totalPerplexity += perplexity
        totalEmotion += emotionStrength
        totalReadability += readability
        totalRichness += richness
        totalSentLength += sentLength
        totalGrammarIssues += grammarIssueCount

        aiFlags = 0
        if perplexity < 50: aiFlags += 1
        if emotionStrength < 0.4: aiFlags += 1
        if readability > 60: aiFlags += 1
        if richness < 0.5: aiFlags += 1
        if grammarIssueCount == 0: aiFlags += 1

        isAi = aiFlags >= 4
        if isAi:
            aiCount += 1
        else:
            humanCount += 1

        results.append({
            'sentence': sentence,
            'perplexity': round(perplexity, 2),
            'emotionStrength': round(emotionStrength, 2),
            'readability': round(readability, 2),
            'lexicalRichness': round(richness, 2),
            'sentenceLength': sentLength,
            'grammarIssues': grammarIssueCount,
            'aiFlags': aiFlags,
            'isAi': isAi
        })

    aiPercentage = (aiCount / len(sentences)) * 100
    avgPerplexity = totalPerplexity / len(sentences)
    avgEmotion = totalEmotion / len(sentences)
    avgReadability = totalReadability / len(sentences)
    avgRichness = totalRichness / len(sentences)
    avgSentLength = totalSentLength / len(sentences)
    avgGrammarIssues = totalGrammarIssues / len(sentences)

    verdict = "PLAGIARISM DETECTED" if aiPercentage > 15 else "No plagiarism"

    return results, aiPercentage, verdict

<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class LessonText extends Model
{
    use HasFactory;

    protected $fillable = [
        'lesson_id',
        'language',
        'format',
        'content',
        'reading_time',
        'is_primary',
        'meta',
    ];

    protected function casts(): array
    {
        return [
            'is_primary' => 'boolean',
            'meta' => 'array',
        ];
    }

    // Relationships
    public function lesson()
    {
        return $this->belongsTo(Lesson::class);
    }

    /**
     * Разбить текст на предложения для субтитров
     */
    public function getSentences(): array
    {
        $content = $this->content;
        
        // Удаляем markdown разметку для чистого текста
        $cleanText = strip_tags(str_replace(['#', '*', '_', '`'], '', $content));
        
        // Разбиваем на предложения по знакам препинания
        $sentences = preg_split('/[.!?]+/', $cleanText, -1, PREG_SPLIT_NO_EMPTY);
        
        $result = [];
        foreach ($sentences as $index => $sentence) {
            $sentence = trim($sentence);
            if (!empty($sentence)) {
                $result[] = [
                    'id' => $index + 1,
                    'text' => $sentence . '.', // Добавляем точку обратно
                    'duration' => $this->estimateSentenceDuration($sentence)
                ];
            }
        }
        
        return $result;
    }

    /**
     * Оценить продолжительность произнесения предложения (в секундах)
     */
    private function estimateSentenceDuration(string $sentence): int
    {
        // Примерно 3-4 слова в секунду для жестового языка
        $wordCount = str_word_count($sentence);
        return max(2, ceil($wordCount / 3)); // Минимум 2 секунды на предложение
    }

    /**
     * Получить полный текст урока
     */
    public function getFullText(): string
    {
        return strip_tags(str_replace(['#', '*', '_', '`'], '', $this->content));
    }
}
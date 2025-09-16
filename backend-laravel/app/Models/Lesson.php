<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Lesson extends Model
{
    use HasFactory;

    protected $fillable = [
        'title',
        'description',
        'module_id',
        'order_index',
        'type',
        'content',
        'video_url',
        'gesture_data',
        'duration_minutes',
        'is_published',
    ];

    protected function casts(): array
    {
        return [
            'gesture_data' => 'array',
            'is_published' => 'boolean',
        ];
    }

    // Relationships
    public function module()
    {
        return $this->belongsTo(Module::class);
    }

    public function progress()
    {
        return $this->hasMany(Progress::class);
    }

    public function texts()
    {
        return $this->hasMany(LessonText::class);
    }

    public function primaryText()
    {
        return $this->hasOne(LessonText::class)->where('is_primary', true);
    }
}

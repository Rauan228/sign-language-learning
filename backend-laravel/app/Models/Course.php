<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Course extends Model
{
    use HasFactory;

    protected $fillable = [
        'title',
        'description',
        'image',
        'price',
        'difficulty_level',
        'duration_hours',
        'is_published',
        'instructor_id',
        'tags',
        'enrollment_count',
        'rating',
    ];

    protected function casts(): array
    {
        return [
            'price' => 'decimal:2',
            'rating' => 'decimal:2',
            'is_published' => 'boolean',
            'tags' => 'array',
        ];
    }

    // Relationships
    public function instructor()
    {
        return $this->belongsTo(User::class, 'instructor_id');
    }

    public function modules()
    {
        return $this->hasMany(Module::class)->orderBy('order_index');
    }

    public function purchases()
    {
        return $this->hasMany(Purchase::class);
    }

    public function progress()
    {
        return $this->hasMany(Progress::class);
    }

    public function certificates()
    {
        return $this->hasMany(Certificate::class);
    }

    public function students()
    {
        return $this->belongsToMany(User::class, 'purchases')
                    ->wherePivot('status', 'completed');
    }
}

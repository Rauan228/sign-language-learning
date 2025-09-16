<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Test extends Model
{
    use HasFactory;

    protected $fillable = [
        'title',
        'description',
        'module_id',
        'questions',
        'passing_score',
        'time_limit_minutes',
        'is_published',
    ];

    protected function casts(): array
    {
        return [
            'questions' => 'array',
            'is_published' => 'boolean',
        ];
    }

    // Relationships
    public function module()
    {
        return $this->belongsTo(Module::class);
    }
}

<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\Course;
use App\Models\Module;
use App\Models\Lesson;
use App\Models\User;
use Illuminate\Support\Facades\DB;

class PhysicsCourseSeeder extends Seeder
{
    public function run(): void
    {
        // Set UTF-8 encoding for MySQL connection
        DB::statement('SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci');
        
        // Create instructor user
        $instructor = User::firstOrCreate(
            ['email' => 'physics.teacher@example.com'],
            [
                'name' => 'Physics Teacher',
                'password' => bcrypt('password'),
            ]
        );

        // Create Physics Course for Grade 6
        $course = Course::create([
            'title' => 'Physics for Grade 6',
            'description' => 'Complete physics course for 6th grade students covering fundamental concepts',
            'price' => 0.00,
            'difficulty_level' => 'beginner',
            'duration_hours' => 36,
            'is_published' => true,
            'instructor_id' => $instructor->id,
            'tags' => json_encode(['physics', 'grade-6', 'science', 'education']),
            'enrollment_count' => 0,
            'rating' => 0
        ]);

        // Module 1: Introduction to Physics
        $module1 = Module::create([
            'title' => 'Introduction to Physics',
            'description' => 'Basic concepts and what physics studies',
            'course_id' => $course->id,
            'order_index' => 1,
            'is_published' => true,
            'duration_minutes' => 360
        ]);

        // Lessons for Module 1
        $lessons1 = [
            [
                'title' => 'What is Physics',
                'description' => 'Introduction to physics as a science',
                'content' => 'Physics is the science that studies nature and natural phenomena around us.',
                'video_url' => 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                'duration_minutes' => 8
            ],
            [
                'title' => 'Physical Phenomena',
                'description' => 'Examples of physical phenomena in everyday life',
                'content' => 'Physical phenomena are events that occur in nature and can be observed.',
                'video_url' => 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                'duration_minutes' => 10
            ]
        ];

        foreach ($lessons1 as $index => $lessonData) {
            Lesson::create(array_merge($lessonData, [
                'module_id' => $module1->id,
                'order_index' => $index + 1,
                'type' => 'video',
                'is_published' => true
            ]));
        }

        // Module 2: Matter and Its Properties
        $module2 = Module::create([
            'title' => 'Matter and Its Properties',
            'description' => 'Study of matter states and properties',
            'course_id' => $course->id,
            'order_index' => 2,
            'is_published' => true,
            'duration_minutes' => 420
        ]);

        // Lessons for Module 2
        $lessons2 = [
            [
                'title' => 'States of Matter',
                'description' => 'Solid, liquid, and gas states',
                'content' => 'Matter exists in three main states: solid, liquid, and gas.',
                'video_url' => 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                'duration_minutes' => 12
            ],
            [
                'title' => 'Properties of Solids',
                'description' => 'Characteristics of solid matter',
                'content' => 'Solids have definite shape and volume.',
                'video_url' => 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                'duration_minutes' => 10
            ]
        ];

        foreach ($lessons2 as $index => $lessonData) {
            Lesson::create(array_merge($lessonData, [
                'module_id' => $module2->id,
                'order_index' => $index + 1,
                'type' => 'video',
                'is_published' => true
            ]));
        }

        echo "Physics course for Grade 6 has been successfully seeded!\n";
    }
}
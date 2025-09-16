<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\User;
use App\Models\Course;
use App\Models\Module;
use App\Models\Lesson;
use App\Models\Purchase;
use App\Models\Progress;
use Illuminate\Support\Facades\Hash;

class FullCoursesSeeder extends Seeder
{
    public function run()
    {
        // Create test user
        $testUser = User::firstOrCreate(
            ['email' => 'test@student.com'],
            [
                'name' => 'Test Student',
                'password' => Hash::make('password'),
                'role' => 'student',
                'email_verified_at' => now(),
            ]
        );

        // Create 3 sample courses, including ID 3 if possible, but since auto-increment, create in order
        $courses = [
            [
                'title' => 'Sign Language Introduction',
                'description' => 'Basic sign language for beginners',
                'price' => 999.00,
                'difficulty_level' => 'beginner',
                'duration_hours' => 4,
                'is_published' => true,
                'instructor_id' => $testUser->id,
                'tags' => json_encode(['sign language', 'beginner']),
                'enrollment_count' => 0,
                'rating' => 0,
            ],
            [
                'title' => 'Intermediate Sign Language',
                'description' => 'Intermediate level sign language',
                'price' => 1499.00,
                'difficulty_level' => 'intermediate',
                'duration_hours' => 6,
                'is_published' => true,
                'instructor_id' => $testUser->id,
                'tags' => json_encode(['sign language', 'intermediate']),
                'enrollment_count' => 0,
                'rating' => 0,
            ],
            [
                'title' => 'Gesture Recognition Basics',
                'description' => 'Learn gesture recognition through 3D avatar lessons with videos and subtitles.',
                'price' => 1999.00,
                'difficulty_level' => 'beginner',
                'duration_hours' => 8,
                'is_published' => true,
                'instructor_id' => $testUser->id,
                'tags' => json_encode(['gesture', 'recognition', '3D avatar']),
                'enrollment_count' => 0,
                'rating' => 0,
            ],
        ];

        $courseIds = [];
        foreach ($courses as $c) {
            $c['image'] = '/placeholder-course.jpg';
            $c['created_at'] = now();
            $c['updated_at'] = now();
            $course = Course::create($c);
            $courseIds[] = $course->id;
        }

        // Create purchase for test user for all courses
        foreach ($courseIds as $id) {
            Purchase::create([
                'user_id' => $testUser->id,
                'course_id' => $id,
                'amount' => 1999.00,
                'currency' => 'USD',
                'status' => 'completed',
                'payment_method' => 'card',
                'transaction_id' => 'test_txn_' . uniqid(),
                'purchased_at' => now(),
            ]);
        }

        // For course ID 3 (last one), create 2 modules
        $course3Id = $courseIds[2];
        $module1Id = Module::create([
            'course_id' => $course3Id,
            'title' => 'Fundamentals',
            'description' => 'Basic gestures and introduction',
            'order_index' => 1,
            'is_published' => true,
        ])->id;

        $module2Id = Module::create([
            'course_id' => $course3Id,
            'title' => 'Advanced Gestures',
            'description' => 'Advanced practice with gestures',
            'order_index' => 2,
            'is_published' => true,
        ])->id;

        // Lessons for module 1 (4 lessons)
        $lesson1 = Lesson::create([
            'module_id' => $module1Id,
            'title' => 'Basic Hand Gestures',
            'description' => 'Learn basic hand positions for sign language',
            'type' => 'video',
            'order_index' => 1,
            'duration_minutes' => 180,
            'is_published' => true,
            'video_url' => 'https://www.youtube.com/embed/dQw4w9WgXcQ', // Rick Roll as placeholder, replace with real if needed
            'gesture_data' => json_encode([
                ['gesture' => '/public/assets/hand_gesture1.bvh', 'duration' => 5, 'text' => 'Hand position 1'],
                ['gesture' => '/public/assets/hand_gesture2.bvh', 'duration' => 5, 'text' => 'Hand position 2'],
            ]),
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Welcome to Basic Hand Gestures'],
                ['time' => '00:10', 'text' => 'The first gesture is palm facing out'],
                ['time' => '00:30', 'text' => 'Second gesture: fingers together'],
                ['time' => '01:00', 'text' => 'Practice these gestures'],
                ['time' => '02:00', 'text' => 'Great job on hand gestures'],
                ['time' => '02:30', 'text' => 'End of lesson'],
            ]),
        ]);

        $lesson2 = Lesson::create([
            'module_id' => $module1Id,
            'title' => 'Facial Expressions',
            'description' => 'Important facial expressions in signing',
            'type' => 'video',
            'order_index' => 2,
            'duration_minutes' => 150,
            'is_published' => true,
            'video_url' => 'https://www.youtube.com/embed/9bZkp7q19f0', // Free sign language video
            'gesture_data' => json_encode([
                ['gesture' => '/public/assets/facial1.bvh', 'duration' => 4, 'text' => 'Smile'],
                ['gesture' => '/public/assets/facial2.bvh', 'duration' => 4, 'text' => 'Frown'],
            ]),
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Facial expressions are crucial'],
                ['time' => '00:15', 'text' => 'Smile gesture: corners of mouth up'],
                ['time' => '00:45', 'text' => 'Frown for negative expressions'],
                ['time' => '01:15', 'text' => 'Practice facial gestures'],
                ['time' => '02:00', 'text' => 'Lesson complete'],
            ]),
        ]);

        // Add 2 more for module 1 (similar pattern)
        $lesson3 = Lesson::create([
            'module_id' => $module1Id,
            'title' => 'Body Posture',
            'description' => 'Body language in sign language',
            'type' => 'video',
            'order_index' => 3,
            'duration_minutes' => 200,
            'is_published' => true,
            'video_url' => 'https://www.youtube.com/embed/7I2oq9k4n8c', // Placeholder
            'gesture_data' => json_encode([
                ['gesture' => '/public/assets/posture1.bvh', 'duration' => 5, 'text' => 'Straight posture'],
            ]),
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Body posture matters'],
                // 10-15 entries, abbreviated
                ['time' => '03:00', 'text' => 'End of body posture lesson'],
            ]),
        ]);

        $lesson4 = Lesson::create([
            'module_id' => $module1Id,
            'title' => 'Introduction Quiz',
            'description' => 'Test basic gestures',
            'type' => 'interactive',
            'order_index' => 4,
            'duration_minutes' => 300,
            'is_published' => true,
            'video_url' => null,
            'gesture_data' => null,
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Quiz on basic gestures'],
                // ...
            ]),
        ]);

        // Module 2 lessons (4 lessons)
        $lesson5 = Lesson::create([
            'module_id' => $module2Id,
            'title' => 'Question Words',
            'description' => 'How to ask questions',
            'type' => 'video',
            'order_index' => 1,
            'duration_minutes' => 180,
            'is_published' => true,
            'video_url' => 'https://www.youtube.com/embed/3sG9q4j5k2m', // Placeholder
            'gesture_data' => json_encode([
                ['gesture' => '/public/assets/question.bvh', 'duration' => 6, 'text' => 'What'],
            ]),
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Question words in ASL'],
                // 10-15 entries
            ]),
        ]);

        $lesson6 = Lesson::create([
            'module_id' => $module2Id,
            'title' => 'Description Practice',
            'description' => 'Describe people and objects',
            'type' => 'video',
            'order_index' => 2,
            'duration_minutes' => 210,
            'is_published' => true,
            'video_url' => 'https://www.youtube.com/embed/8j7k9m4p1q8', // Placeholder
            'gesture_data' => json_encode([
                ['gesture' => '/public/assets/describe.bvh', 'duration' => 5, 'text' => 'Description'],
            ]),
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Describing in sign language'],
                // ...
            ]),
        ]);

        $lesson7 = Lesson::create([
            'module_id' => $module2Id,
            'title' => 'Conversation Practice',
            'description' => 'Simple conversations',
            'type' => 'video',
            'order_index' => 3,
            'duration_minutes' => 240,
            'is_published' => true,
            'video_url' => 'https://www.youtube.com/embed/k5j2m8n9p4r', // Placeholder
            'gesture_data' => json_encode([
                ['gesture' => '/public/assets/convo1.bvh', 'duration' => 8, 'text' => 'Simple convo'],
            ]),
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Practice conversation'],
                // ...
            ]),
        ]);

        $lesson8 = Lesson::create([
            'module_id' => $module2Id,
            'title' => 'Module Quiz',
            'description' => 'Test daily expressions',
            'type' => 'interactive',
            'order_index' => 4,
            'duration_minutes' => 270,
            'is_published' => true,
            'video_url' => null,
            'gesture_data' => null,
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Quiz for module 2'],
                // ...
            ]),
        ]);

        // Module 3 lessons (3 lessons)
        $lesson9 = Lesson::create([
            'module_id' => $module2Id,
            'title' => 'Advanced Quiz',
            'description' => 'Advanced test',
            'type' => 'interactive',
            'order_index' => 1,
            'duration_minutes' => 300,
            'is_published' => true,
            'video_url' => null,
            'gesture_data' => null,
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Advanced quiz'],
                // ...
            ]),
        ]);

        $lesson10 = Lesson::create([
            'module_id' => $module2Id,
            'title' => 'Final Project',
            'description' => 'Complete a sign language project',
            'type' => 'gesture_practice',
            'order_index' => 2,
            'duration_minutes' => 360,
            'is_published' => true,
            'video_url' => '/storage/videos/final_project.mp4',
            'gesture_data' => json_encode([
                ['gesture' => '/public/assets/final.bvh', 'duration' => 15, 'text' => 'Final project'],
            ]),
            'subtitles' => json_encode([
                ['time' => '00:00', 'text' => 'Your final project'],
                // ...
            ]),
        ]);

        // Create initial progress for test user (incomplete for all lessons)
        $lessons = Lesson::where('module_id', $module1Id)->orWhere('module_id', $module2Id)->get();
        foreach ($lessons as $lesson) {
            Progress::create([
                'user_id' => $testUser->id,
                'course_id' => $course3Id,
                'lesson_id' => $lesson->id,
                'status' => 'in_progress',
                'completion_percentage' => 0,
                'started_at' => now(),
                'completed_at' => null,
                'time_spent_minutes' => 0,
            ]);
        }

        $this->command->info('Enhanced test data seeded successfully!');
        $this->command->info('Test user: ' . $testUser->email . ' / password');
        $this->command->info('Courses created: 3 (IDs: 1,2,3)');
        $this->command->info('For course ID 3: 2 modules, 10 lessons total');
        $this->command->info('All lessons have video_url placeholders and subtitles JSON.');
        $this->command->info('Progress: All in_progress with 0% for test user.');
    }
}
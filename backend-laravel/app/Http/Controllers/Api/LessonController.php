<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\Lesson;
use App\Models\LessonText;
use App\Models\Progress;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class LessonController extends Controller
{
    /**
     * Display the specified lesson with subtitles.
     */
    public function show(string $id)
    {
        $lesson = Lesson::with(['module.course.modules.lessons', 'primaryText'])->find($id);
        
        if (!$lesson) {
            return response()->json([
                'success' => false,
                'message' => 'Lesson not found'
            ], 404);
        }

        // Check if user has access to the course
        $user = auth()->user();
        if ($user) {
            $hasAccess = $user->purchases()
                ->where('course_id', $lesson->module->course->id)
                ->where('status', 'completed')
                ->exists();
                
            if (!$hasAccess && !$lesson->module->course->is_free) {
                return response()->json([
                    'success' => false,
                    'message' => 'Access denied. Please purchase the course.'
                ], 403);
            }
        }

        // Get subtitles from lesson_texts table or fallback to content
        $subtitles = [];
        $fullText = '';
        
        if ($lesson->primaryText) {
            $sentences = $lesson->primaryText->getSentences();
            $currentTime = 0;
            
            foreach ($sentences as $sentence) {
                $duration = $sentence['duration'] ?? 5;
                $subtitles[] = [
                    'id' => (string)$sentence['id'],
                    'start' => $currentTime,
                    'end' => $currentTime + $duration,
                    'text' => $sentence['text'],
                    'language' => 'ru',
                    'speaker' => null,
                    'position' => 'bottom'
                ];
                $currentTime += $duration;
            }
            
            $fullText = $lesson->primaryText->getFullText();
        } else {
            // Fallback to old method if no lesson_texts exist
            $baseTime = 0;
            $sentences = preg_split('/[.!?]+/', $lesson->content);
            foreach ($sentences as $index => $sentence) {
                $sentence = trim($sentence);
                if (!empty($sentence)) {
                    $subtitles[] = [
                        'id' => (string)($index + 1),
                        'start' => $baseTime,
                        'end' => $baseTime + 5,
                        'text' => $sentence . '.',
                        'language' => 'ru',
                        'speaker' => null,
                        'position' => 'bottom'
                    ];
                    $baseTime += 5;
                }
            }
            $fullText = strip_tags($lesson->content);
        }

        // Add title as first subtitle
        array_unshift($subtitles, [
            'id' => '0',
            'start' => 0,
            'end' => 3,
            'text' => $lesson->title,
            'language' => 'ru',
            'speaker' => null,
            'position' => 'bottom'
        ]);

        return response()->json([
            'success' => true,
            'data' => [
                'lesson' => $lesson,
                'subtitles' => $subtitles,
                'fullText' => $fullText,
                'gesture_data' => $lesson->gesture_data,
                'video_url' => $lesson->video_url ?: '/placeholder-video.mp4'
            ]
        ]);
    }

    /**
     * Get subtitles for a specific lesson.
     */
    public function getSubtitles(string $id)
    {
        $lesson = Lesson::with('primaryText')->find($id);
        
        if (!$lesson) {
            return response()->json([
                'success' => false,
                'message' => 'Lesson not found'
            ], 404);
        }

        // Check if user has access to the course
        $user = auth()->user();
        if ($user) {
            $hasAccess = $user->purchases()
                ->where('course_id', $lesson->module->course->id)
                ->where('status', 'completed')
                ->exists();
                
            if (!$hasAccess && !$lesson->module->course->is_free) {
                return response()->json([
                    'success' => false,
                    'message' => 'Access denied. Please purchase the course.'
                ], 403);
            }
        }

        $subtitles = [];
        $fullText = '';
        
        if ($lesson->primaryText) {
            $sentences = $lesson->primaryText->getSentences();
            $currentTime = 0;
            
            foreach ($sentences as $sentence) {
                $duration = $sentence['duration'] ?? 5;
                $subtitles[] = [
                    'id' => (string)$sentence['id'],
                    'start' => $currentTime,
                    'end' => $currentTime + $duration,
                    'text' => $sentence['text'],
                    'language' => 'ru',
                    'speaker' => null,
                    'position' => 'bottom'
                ];
                $currentTime += $duration;
            }
            
            $fullText = $lesson->primaryText->getFullText();
        } else {
            // Fallback to old method if no lesson_texts exist
            $baseTime = 0;
            $sentences = preg_split('/[.!?]+/', $lesson->content);
            foreach ($sentences as $index => $sentence) {
                $sentence = trim($sentence);
                if (!empty($sentence)) {
                    $subtitles[] = [
                        'id' => (string)($index + 1),
                        'start' => $baseTime,
                        'end' => $baseTime + 5,
                        'text' => $sentence . '.',
                        'language' => 'ru',
                        'speaker' => null,
                        'position' => 'bottom'
                    ];
                    $baseTime += 5;
                }
            }
            $fullText = strip_tags($lesson->content);
        }

        return response()->json([
            'success' => true,
            'data' => [
                'subtitles' => $subtitles,
                'fullText' => $fullText
            ]
        ]);
    }

    /**
     * Mark lesson as completed.
     */
    public function markComplete(Request $request, string $id)
    {
        $user = $request->user();
        
        if (!$user) {
            return response()->json([
                'success' => false,
                'message' => 'Unauthorized'
            ], 401);
        }

        $lesson = Lesson::with('module')->find($id);
        
        if (!$lesson) {
            return response()->json([
                'success' => false,
                'message' => 'Lesson not found'
            ], 404);
        }

        // Update or create progress record
        $progress = Progress::updateOrCreate(
            [
                'user_id' => $user->id,
                'course_id' => $lesson->module->course_id,
                'lesson_id' => $lesson->id,
            ],
            [
                'status' => 'completed',
                'completion_percentage' => 100,
                'completed_at' => now(),
                'started_at' => now()
            ]
        );

        return response()->json([
            'success' => true,
            'message' => 'Lesson marked as completed',
            'data' => $progress
        ]);
    }

    /**
     * Get lesson progress for current user.
     */
    public function getProgress(Request $request, string $id)
    {
        $user = $request->user();
        
        if (!$user) {
            return response()->json([
                'success' => false,
                'message' => 'Unauthorized'
            ], 401);
        }

        $lesson = Lesson::with('module')->find($id);
        
        if (!$lesson) {
            return response()->json([
                'success' => false,
                'message' => 'Lesson not found'
            ], 404);
        }

        // Get progress record
        $progress = Progress::where('user_id', $user->id)
            ->where('lesson_id', $id)
            ->where('course_id', $lesson->module->course_id)
            ->first();

        if (!$progress) {
            // Create initial progress record
            $progress = Progress::create([
                'user_id' => $user->id,
                'course_id' => $lesson->module->course_id,
                'lesson_id' => $id,
                'status' => 'not_started',
                'completion_percentage' => 0,
                'started_at' => null,
                'completed_at' => null,
                'time_spent_minutes' => 0,
                'watched_duration' => 0,
                'is_completed' => false,
            ]);
        }

        return response()->json([
            'success' => true,
            'data' => [
                'id' => $progress->id,
                'lessonId' => (int)$id,
                'watchedDuration' => $progress->watched_duration ?? 0,
                'isCompleted' => $progress->is_completed ?? false,
                'completedAt' => $progress->completed_at ? $progress->completed_at->toISOString() : null,
                'lastWatchedAt' => $progress->updated_at->toISOString(),
                'progress_percentage' => $progress->completion_percentage,
                'time_spent' => $progress->time_spent_minutes * 60,
                'completed' => $progress->status === 'completed',
                'sessions' => [] // Mock sessions for now
            ]
        ]);
    }

    /**
     * Save lesson progress.
     */
    public function saveProgress(Request $request, string $id)
    {
        $user = $request->user();
        
        if (!$user) {
            return response()->json([
                'success' => false,
                'message' => 'Unauthorized'
            ], 401);
        }

        $lesson = Lesson::with('module')->find($id);
        
        if (!$lesson) {
            return response()->json([
                'success' => false,
                'message' => 'Lesson not found'
            ], 404);
        }

        $validator = Validator::make($request->all(), [
            'watchedDuration' => 'required|integer|min:0',
            'isCompleted' => 'boolean',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation errors',
                'errors' => $validator->errors()
            ], 422);
        }

        $watchedDuration = $request->input('watchedDuration', 0);
        $isCompleted = $request->input('isCompleted', false);
        $timeSpentMinutes = ceil($watchedDuration / 60);

        // Update or create progress record
        $progress = Progress::updateOrCreate(
            [
                'user_id' => $user->id,
                'course_id' => $lesson->module->course_id,
                'lesson_id' => $id,
            ],
            [
                'status' => $isCompleted ? 'completed' : 'in_progress',
                'completion_percentage' => $isCompleted ? 100 : min(90, ceil(($watchedDuration / 600) * 100)), // Assume 10min videos
                'time_spent_minutes' => $timeSpentMinutes,
                'watched_duration' => $watchedDuration,
                'is_completed' => $isCompleted,
                'started_at' => now(),
                'completed_at' => $isCompleted ? now() : null,
            ]
        );

        return response()->json([
            'success' => true,
            'message' => 'Progress saved successfully',
            'data' => [
                'id' => $progress->id,
                'lessonId' => (int)$id,
                'watchedDuration' => $progress->watched_duration,
                'isCompleted' => $progress->is_completed,
                'completedAt' => $progress->completed_at ? $progress->completed_at->toISOString() : null,
                'lastWatchedAt' => $progress->updated_at->toISOString(),
                'progress_percentage' => $progress->completion_percentage,
                'time_spent' => $progress->time_spent_minutes * 60,
                'completed' => $progress->status === 'completed'
            ]
        ]);
    }

    /**
     * Save watching session.
     */
    public function saveSession(Request $request, string $id)
    {
        $user = $request->user();
        
        if (!$user) {
            return response()->json([
                'success' => false,
                'message' => 'Unauthorized'
            ], 401);
        }

        $lesson = Lesson::with('module')->find($id);
        
        if (!$lesson) {
            return response()->json([
                'success' => false,
                'message' => 'Lesson not found'
            ], 404);
        }

        $validator = Validator::make($request->all(), [
            'startTime' => 'required|integer|min:0',
            'endTime' => 'required|integer|min:0',
            'duration' => 'required|integer|min:0',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation errors',
                'errors' => $validator->errors()
            ], 422);
        }

        // For now, just update the progress with session info
        $duration = $request->input('duration', 0);
        $timeSpentMinutes = ceil($duration / 60);

        $progress = Progress::updateOrCreate(
            [
                'user_id' => $user->id,
                'course_id' => $lesson->module->course_id,
                'lesson_id' => $id,
            ],
            [
                'status' => 'in_progress',
                'time_spent_minutes' => $timeSpentMinutes,
                'started_at' => now(),
            ]
        );

        return response()->json([
            'success' => true,
            'message' => 'Session saved successfully',
            'data' => [
                'sessionId' => uniqid(),
                'lessonId' => (int)$id,
                'startTime' => $request->input('startTime'),
                'endTime' => $request->input('endTime'),
                'duration' => $duration
            ]
        ]);
    }
}

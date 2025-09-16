<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\Course;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class CourseController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        $courses = Course::with(['modules.lessons'])->get();
        
        return response()->json([
            'success' => true,
            'data' => $courses
        ]);
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'title' => 'required|string|max:255',
            'description' => 'required|string',
            'price' => 'required|numeric|min:0',
            'difficulty_level' => 'required|in:beginner,intermediate,advanced',
            'duration_hours' => 'required|integer|min:1',
            'image' => 'nullable|url',
            'instructor_id' => 'required|exists:users,id',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation errors',
                'errors' => $validator->errors()
            ], 422);
        }

        $course = Course::create($request->all());

        return response()->json([
            'success' => true,
            'message' => 'Course created successfully',
            'data' => $course
        ], 201);
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        $course = Course::with(['modules.lessons'])->find($id);
        
        if (!$course) {
            return response()->json([
                'success' => false,
                'message' => 'Course not found'
            ], 404);
        }

        return response()->json([
            'success' => true,
            'data' => $course
        ]);
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        $course = Course::find($id);
        
        if (!$course) {
            return response()->json([
                'success' => false,
                'message' => 'Course not found'
            ], 404);
        }

        $validator = Validator::make($request->all(), [
            'title' => 'sometimes|required|string|max:255',
            'description' => 'sometimes|required|string',
            'price' => 'sometimes|required|numeric|min:0',
            'difficulty_level' => 'sometimes|required|in:beginner,intermediate,advanced',
            'duration_hours' => 'sometimes|required|integer|min:1',
            'image' => 'nullable|url',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation errors',
                'errors' => $validator->errors()
            ], 422);
        }

        $course->update($request->all());

        return response()->json([
            'success' => true,
            'message' => 'Course updated successfully',
            'data' => $course
        ]);
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        $course = Course::find($id);
        
        if (!$course) {
            return response()->json([
                'success' => false,
                'message' => 'Course not found'
            ], 404);
        }

        $course->delete();

        return response()->json([
            'success' => true,
            'message' => 'Course deleted successfully'
        ]);
    }

    /**
     * Get courses enrolled by the authenticated user.
     */
    public function enrolled(Request $request)
    {
        $user = $request->user();
        $page = $request->get('page', 1);
        $pageSize = $request->get('page_size', 12);
        
        // Get courses through purchases relationship with pagination
        $coursesQuery = Course::whereHas('purchases', function ($query) use ($user) {
            $query->where('user_id', $user->id)
                  ->where('status', 'completed');
        })->with(['modules.lessons', 'purchases' => function ($query) use ($user) {
            $query->where('user_id', $user->id)->where('status', 'completed');
        }]);
        
        $totalCourses = $coursesQuery->count();
        $courses = $coursesQuery->skip(($page - 1) * $pageSize)
                               ->take($pageSize)
                               ->get();
        
        // Transform courses data to include enrollment info
        $transformedCourses = $courses->map(function ($course) use ($user) {
            $purchase = $course->purchases->first();
            
            // Calculate real progress based on completed lessons
            $totalLessons = \App\Models\Lesson::whereHas('module', function($query) use ($course) {
                $query->where('course_id', $course->id);
            })->count();
            
            $completedLessons = \App\Models\Progress::where('user_id', $user->id)
                ->where('course_id', $course->id)
                ->where('status', 'completed')
                ->count();
                
            $progress = $totalLessons > 0 ? round(($completedLessons / $totalLessons) * 100) : 0;
            
            return [
                'id' => $course->id,
                'title' => $course->title,
                'description' => $course->description,
                'thumbnail_url' => $course->thumbnail_url,
                'price' => $course->price,
                'original_price' => $course->original_price,
                'rating' => $course->rating ?? 4.5,
                'students_count' => $course->enrollment_count ?? 0,
                'duration_hours' => $course->duration_hours ?? 10,
                'level' => $course->level,
                'category' => $course->category,
                'teacher_name' => $course->instructor ?? 'Instructor',
                'is_free' => $course->is_free ?? false,
                'is_new' => $course->is_new ?? false,
                'is_bestseller' => $course->is_bestseller ?? false,
                'enrollment_date' => $purchase ? $purchase->purchased_at->format('Y-m-d H:i:s') : null,
                'progress' => $progress
            ];
        });
        
        return response()->json([
            'success' => true,
            'data' => $transformedCourses,
            'total' => $totalCourses,
            'page' => (int) $page,
            'page_size' => (int) $pageSize,
            'total_pages' => ceil($totalCourses / $pageSize)
        ]);
    }

    /**
     * Purchase a course.
     */
    public function purchase(Request $request, string $id)
    {
        $user = $request->user();
        $course = Course::find($id);
        
        if (!$course) {
            return response()->json([
                'success' => false,
                'message' => 'Course not found'
            ], 404);
        }

        // Check if user already purchased this course
        $existingPurchase = \App\Models\Purchase::where('user_id', $user->id)
            ->where('course_id', $course->id)
            ->where('status', 'completed')
            ->first();

        if ($existingPurchase) {
            return response()->json([
                'success' => false,
                'message' => 'You have already purchased this course'
            ], 400);
        }

        $validator = Validator::make($request->all(), [
            // Allow additional methods used by the frontend
            'payment_method' => 'required|string|in:card,paypal,bank_transfer,yandex,qiwi,free',
            'card_number' => 'required_if:payment_method,card|string',
            'card_holder' => 'required_if:payment_method,card|string',
            'expiry_date' => 'required_if:payment_method,card|string',
            'cvv' => 'required_if:payment_method,card|string|size:3'
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation errors',
                'errors' => $validator->errors()
            ], 422);
        }

        // Additional rule: 'free' method only allowed for free courses
        if ($request->payment_method === 'free' && !($course->is_free || (float) $course->price <= 0)) {
            return response()->json([
                'success' => false,
                'message' => 'Free payment method is only allowed for free courses',
                'errors' => ['payment_method' => ['Free method is not allowed for paid courses']]
            ], 422);
        }

        // Simulate payment processing (in real app, integrate with payment gateway)
        $transactionId = 'txn_' . uniqid();
        
        // Create purchase record
        $purchase = \App\Models\Purchase::create([
            'user_id' => $user->id,
            'course_id' => $course->id,
            'amount' => $course->price,
            'currency' => 'RUB',
            'status' => 'completed', // In real app, this would be 'pending' initially
            'payment_method' => $request->payment_method,
            'transaction_id' => $transactionId,
            'purchased_at' => now()
        ]);

        // Update course enrollment count
        $course->increment('enrollment_count');

        return response()->json([
            'success' => true,
            'message' => 'Course purchased successfully',
            'data' => [
                'purchase_id' => $purchase->id,
                'transaction_id' => $transactionId,
                'course' => $course,
                'amount' => $purchase->amount,
                'currency' => $purchase->currency
            ]
        ], 201);
    }

    /**
     * Check if user has access to the course
     */
    public function checkAccess($courseId)
    {
        $user = auth()->user();
        
        if (!$user) {
            return response()->json([
                'success' => false,
                'message' => 'Unauthorized',
                'has_access' => false
            ], 401);
        }

        $course = Course::find($courseId);
        
        if (!$course) {
            return response()->json([
                'success' => false,
                'message' => 'Course not found',
                'has_access' => false
            ], 404);
        }

        // Check if course is free
        if ($course->is_free) {
            return response()->json([
                'success' => true,
                'has_access' => true,
                'access_type' => 'free'
            ]);
        }

        // Check if user has purchased the course
        $purchase = $user->purchases()
            ->where('course_id', $courseId)
            ->where('status', 'completed')
            ->first();

        if ($purchase) {
            return response()->json([
                'success' => true,
                'has_access' => true,
                'access_type' => 'purchased',
                'purchase_date' => $purchase->purchased_at
            ]);
        }

        return response()->json([
            'success' => false,
            'message' => 'Access denied. Please purchase the course.',
            'has_access' => false
        ], 403);
    }

    /**
     * Get user's progress for a specific course
     */
    public function getProgress($courseId)
    {
        $user = auth()->user();
        
        if (!$user) {
            return response()->json([
                'success' => false,
                'message' => 'Unauthorized'
            ], 401);
        }

        $course = Course::with(['modules.lessons'])->find($courseId);
        
        if (!$course) {
            return response()->json([
                'success' => false,
                'message' => 'Course not found'
            ], 404);
        }

        // Get all lesson IDs for this course
        $allLessonIds = [];
        foreach ($course->modules as $module) {
            foreach ($module->lessons as $lesson) {
                $allLessonIds[] = $lesson->id;
            }
        }

        // Get all progress records for this course
        $progressRecords = \App\Models\Progress::where('user_id', $user->id)
            ->where('course_id', $courseId)
            ->get()
            ->keyBy('lesson_id');
        
        $completedLessons = $progressRecords->where('status', 'completed')->pluck('lesson_id')->toArray();
        
        $totalLessons = count($allLessonIds);
        $completedCount = count($completedLessons);
        $progressPercentage = $totalLessons > 0 ? round(($completedCount / $totalLessons) * 100) : 0;

        return response()->json([
            'success' => true,
            'data' => $course->modules->map(function ($module) use ($completedLessons, $progressRecords) {
                $moduleLessons = $module->lessons;
                $moduleCompletedLessons = $moduleLessons->filter(function ($lesson) use ($completedLessons) {
                    return in_array($lesson->id, $completedLessons);
                })->count();
                
                $lessonsProgress = $moduleLessons->map(function ($lesson) use ($progressRecords) {
                    $progress = $progressRecords->get($lesson->id);
                    return [
                        'lessonId' => $lesson->id,
                        'watchedDuration' => $progress ? $progress->watched_duration : 0,
                        'totalDuration' => $lesson->duration,
                        'progressPercentage' => $progress ? $progress->completion_percentage : 0,
                        'isCompleted' => $progress ? $progress->is_completed : false,
                        'lastWatchedAt' => $progress ? $progress->updated_at->toISOString() : null
                    ];
                });
                
                return [
                    'moduleId' => $module->id,
                    'completedLessons' => $moduleCompletedLessons,
                    'totalLessons' => $moduleLessons->count(),
                    'completedLessonIds' => $moduleLessons->filter(function ($lesson) use ($completedLessons) {
                        return in_array($lesson->id, $completedLessons);
                    })->pluck('id')->toArray(),
                    'progressPercentage' => $moduleLessons->count() > 0 ? round(($moduleCompletedLessons / $moduleLessons->count()) * 100) : 0,
                    'lessonsProgress' => $lessonsProgress
                ];
            })
        ]);
    }

    /**
     * Mark a lesson as completed
     */
    public function markLessonComplete($courseId, $lessonId)
    {
        $user = auth()->user();
        
        if (!$user) {
            return response()->json([
                'success' => false,
                'message' => 'Unauthorized'
            ], 401);
        }

        // Check if user has access to the course
        $accessCheck = $this->checkAccess($courseId);
        $accessData = json_decode($accessCheck->getContent(), true);
        
        if (!$accessData['has_access']) {
            return response()->json([
                'success' => false,
                'message' => 'Access denied'
            ], 403);
        }

        // In a real implementation, you would save this to a lesson_progress table
        // For now, we'll just return success
        return response()->json([
            'success' => true,
            'message' => 'Lesson marked as completed',
            'data' => [
                'course_id' => $courseId,
                'lesson_id' => $lessonId,
                'completed_at' => now()
            ]
        ]);
    }
}

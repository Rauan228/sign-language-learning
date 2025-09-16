<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\Progress;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class ProgressController extends Controller
{
    /**
     * Display user's progress for a specific course.
     */
    public function index(Request $request)
    {
        $courseId = $request->query('course_id');
        $userId = $request->user()->id;
        
        $query = Progress::with(['course', 'lesson'])
            ->where('user_id', $userId);
            
        if ($courseId) {
            $query->where('course_id', $courseId);
        }
        
        $progress = $query->get();
        
        return response()->json([
            'success' => true,
            'data' => $progress
        ]);
    }

    /**
     * Store or update lesson progress.
     */
    public function store(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'course_id' => 'required|exists:courses,id',
            'lesson_id' => 'required|exists:lessons,id',
            'is_completed' => 'required|boolean',
            'completion_percentage' => 'required|integer|min:0|max:100',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation errors',
                'errors' => $validator->errors()
            ], 422);
        }

        $progress = Progress::updateOrCreate(
            [
                'user_id' => $request->user()->id,
                'course_id' => $request->course_id,
                'lesson_id' => $request->lesson_id,
            ],
            [
                'is_completed' => $request->is_completed,
                'completion_percentage' => $request->completion_percentage,
                'completed_at' => $request->is_completed ? now() : null,
            ]
        );

        return response()->json([
            'success' => true,
            'message' => 'Progress updated successfully',
            'data' => $progress->load(['course', 'lesson'])
        ]);
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        $progress = Progress::with(['course', 'lesson', 'user'])->find($id);
        
        if (!$progress) {
            return response()->json([
                'success' => false,
                'message' => 'Progress not found'
            ], 404);
        }

        return response()->json([
            'success' => true,
            'data' => $progress
        ]);
    }

    /**
     * Get course completion statistics.
     */
    public function courseStats(Request $request, $courseId)
    {
        $userId = $request->user()->id;
        
        $totalLessons = \App\Models\Lesson::whereHas('module', function($query) use ($courseId) {
            $query->where('course_id', $courseId);
        })->count();
        
        $completedLessons = Progress::where('user_id', $userId)
            ->where('course_id', $courseId)
            ->where('is_completed', true)
            ->count();
            
        $overallProgress = $totalLessons > 0 ? round(($completedLessons / $totalLessons) * 100, 2) : 0;
        
        return response()->json([
            'success' => true,
            'data' => [
                'course_id' => $courseId,
                'total_lessons' => $totalLessons,
                'completed_lessons' => $completedLessons,
                'overall_progress' => $overallProgress
            ]
        ]);
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        $progress = Progress::find($id);
        
        if (!$progress) {
            return response()->json([
                'success' => false,
                'message' => 'Progress not found'
            ], 404);
        }

        $validator = Validator::make($request->all(), [
            'is_completed' => 'sometimes|required|boolean',
            'completion_percentage' => 'sometimes|required|integer|min:0|max:100',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation errors',
                'errors' => $validator->errors()
            ], 422);
        }

        $updateData = $request->only(['is_completed', 'completion_percentage']);
        
        if (isset($updateData['is_completed']) && $updateData['is_completed']) {
            $updateData['completed_at'] = now();
        }

        $progress->update($updateData);

        return response()->json([
            'success' => true,
            'message' => 'Progress updated successfully',
            'data' => $progress->load(['course', 'lesson'])
        ]);
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        $progress = Progress::find($id);
        
        if (!$progress) {
            return response()->json([
                'success' => false,
                'message' => 'Progress not found'
            ], 404);
        }

        $progress->delete();

        return response()->json([
            'success' => true,
            'message' => 'Progress deleted successfully'
        ]);
    }

    /**
     * Get user's progress summary statistics.
     */
    public function summary(Request $request)
    {
        $userId = $request->user()->id;
        
        // Get total enrolled courses
        $totalCourses = Progress::where('user_id', $userId)
            ->distinct('course_id')
            ->count('course_id');
            
        // Get completed courses
        $completedCourses = Progress::where('user_id', $userId)
            ->where('status', 'completed')
            ->distinct('course_id')
            ->count('course_id');
            
        // Get total lessons completed
        $completedLessons = Progress::where('user_id', $userId)
            ->where('status', 'completed')
            ->count();
            
        // Get average completion percentage
        $avgCompletion = Progress::where('user_id', $userId)
            ->avg('completion_percentage') ?? 0;
            
        return response()->json([
            'success' => true,
            'data' => [
                'total_courses' => $totalCourses,
                'completed_courses' => $completedCourses,
                'completed_lessons' => $completedLessons,
                'average_completion' => round($avgCompletion, 1)
            ]
        ]);
    }
}

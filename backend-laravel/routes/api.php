<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\AuthController;
use App\Http\Controllers\Api\CourseController;
use App\Http\Controllers\Api\ModuleController;
use App\Http\Controllers\Api\LessonController;
use App\Http\Controllers\Api\ProgressController;

// API v1 routes
Route::prefix('v1')->group(function () {
    // Public routes
    Route::prefix('auth')->group(function () {
        Route::post('/register', [AuthController::class, 'register']);
        Route::post('/login', [AuthController::class, 'login']);
    });
    
    // Public course routes (for browsing)
    Route::get('/courses', [CourseController::class, 'index']);
    
    // Protected routes
    Route::middleware('auth:sanctum')->group(function () {
        // Auth routes
        Route::prefix('auth')->group(function () {
            Route::post('/logout', [AuthController::class, 'logout']);
            Route::get('/user', [AuthController::class, 'user']);
        });
        
        // User enrolled courses (must be before /courses/{id})
        Route::get('/courses/enrolled', [CourseController::class, 'enrolled']);
        
        // Course purchase
        Route::post('/courses/{id}/purchase', [CourseController::class, 'purchase']);
        
        // Course access and progress
        Route::get('/courses/{id}/access', [CourseController::class, 'checkAccess']);
        Route::get('/courses/{id}/progress', [CourseController::class, 'getProgress']);
        Route::post('/courses/{courseId}/lessons/{lessonId}/complete', [CourseController::class, 'markLessonComplete']);
        
        // Course management (admin only - you can add middleware later)
        Route::post('/courses', [CourseController::class, 'store']);
        Route::get('/courses/{id}', [CourseController::class, 'show']);
        Route::put('/courses/{id}', [CourseController::class, 'update']);
        Route::delete('/courses/{id}', [CourseController::class, 'destroy']);
        
        // Module routes
        Route::apiResource('modules', ModuleController::class);
        
        // Lesson routes
        Route::apiResource('lessons', LessonController::class);
        Route::get('/lessons/{id}/subtitles', [LessonController::class, 'getSubtitles']);
        Route::post('/lessons/{id}/complete', [LessonController::class, 'markComplete']);
        Route::get('/lessons/{id}/progress', [LessonController::class, 'getProgress']);
        Route::post('/lessons/{id}/progress', [LessonController::class, 'saveProgress']);
        Route::post('/lessons/{id}/progress/sessions', [LessonController::class, 'saveSession']);
        
        // Progress routes
        Route::get('/progress', [ProgressController::class, 'index']);
        Route::get('/progress/summary', [ProgressController::class, 'summary']);
        Route::post('/progress', [ProgressController::class, 'store']);
        Route::get('/progress/{id}', [ProgressController::class, 'show']);
        Route::put('/progress/{id}', [ProgressController::class, 'update']);
        Route::delete('/progress/{id}', [ProgressController::class, 'destroy']);
        Route::get('/courses/{courseId}/stats', [ProgressController::class, 'courseStats']);
        
        // Gesture recognition endpoint (will be implemented later)
        Route::post('/gesture/recognize', function (Request $request) {
            return response()->json([
                'success' => true,
                'message' => 'Gesture recognition endpoint - to be implemented',
                'data' => null
            ]);
        });
    });
});

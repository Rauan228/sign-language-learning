<?php

require_once 'vendor/autoload.php';

use Illuminate\Foundation\Application;
use Illuminate\Http\Request;
use App\Http\Controllers\Api\LessonController;
use App\Models\User;
use App\Models\Lesson;

// Bootstrap Laravel
$app = require_once 'bootstrap/app.php';
$app->make(Illuminate\Contracts\Console\Kernel::class)->bootstrap();

echo "=== Testing Progress API ===\n";

try {
    // Find a test user and lesson
    $user = User::first();
    $lesson = Lesson::first();
    
    if (!$user || !$lesson) {
        echo "Error: No user or lesson found in database\n";
        exit(1);
    }
    
    echo "\nUsing User ID: {$user->id} ({$user->name})\n";
    echo "Using Lesson ID: {$lesson->id} ({$lesson->title})\n";
    
    // Create a mock request for saving progress
    $request = new Request();
    $request->setUserResolver(function () use ($user) {
        return $user;
    });
    
    // Set request data
    $request->merge([
        'watchedDuration' => 120, // 2 minutes
        'isCompleted' => false
    ]);
    
    echo "\nTesting saveProgress API...\n";
    
    $controller = new LessonController();
    $response = $controller->saveProgress($request, $lesson->id);
    
    echo "Response status: " . $response->getStatusCode() . "\n";
    echo "Response content: " . $response->getContent() . "\n";
    
    // Test getProgress API
    echo "\nTesting getProgress API...\n";
    
    $getRequest = new Request();
    $getRequest->setUserResolver(function () use ($user) {
        return $user;
    });
    
    $getResponse = $controller->getProgress($getRequest, $lesson->id);
    
    echo "Get Response status: " . $getResponse->getStatusCode() . "\n";
    echo "Get Response content: " . $getResponse->getContent() . "\n";
    
} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
    echo "Stack trace: " . $e->getTraceAsString() . "\n";
}

echo "\n=== End Test ===\n";
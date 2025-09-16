<?php

require_once 'vendor/autoload.php';

use Illuminate\Foundation\Application;
use Illuminate\Support\Facades\DB;
use App\Models\Progress;

// Bootstrap Laravel
$app = require_once 'bootstrap/app.php';
$app->make(Illuminate\Contracts\Console\Kernel::class)->bootstrap();

echo "=== Progress Table Check ===\n";

try {
    // Check table structure
    echo "\n1. Table structure:\n";
    $columns = DB::select("DESCRIBE progress");
    foreach ($columns as $column) {
        echo "  {$column->Field} - {$column->Type} - {$column->Null} - {$column->Key}\n";
    }
    
    // Check record count
    echo "\n2. Record count: " . Progress::count() . "\n";
    
    // Show sample records
    echo "\n3. Sample records:\n";
    $samples = Progress::limit(3)->get();
    foreach ($samples as $progress) {
        echo "  ID: {$progress->id}, User: {$progress->user_id}, Lesson: {$progress->lesson_id}, Watched: {$progress->watched_duration}, Completed: " . ($progress->is_completed ? 'Yes' : 'No') . "\n";
    }
    
    // Test creating a progress record
    echo "\n4. Testing Progress model fillable fields:\n";
    $fillable = (new Progress())->getFillable();
    echo "  Fillable: " . implode(', ', $fillable) . "\n";
    
} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
    echo "Stack trace: " . $e->getTraceAsString() . "\n";
}

echo "\n=== End Check ===\n";
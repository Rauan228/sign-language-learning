<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('lessons', function (Blueprint $table) {
            $table->id();
            $table->string('title');
            $table->text('description')->nullable();
            $table->foreignId('module_id')->constrained()->onDelete('cascade');
            $table->integer('order_index')->default(0);
            $table->enum('type', ['video', 'text', 'interactive', 'gesture_practice'])->default('video');
            $table->text('content')->nullable();
            $table->string('video_url')->nullable();
            $table->json('gesture_data')->nullable();
            $table->integer('duration_minutes')->default(0);
            $table->boolean('is_published')->default(false);
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('lessons');
    }
};

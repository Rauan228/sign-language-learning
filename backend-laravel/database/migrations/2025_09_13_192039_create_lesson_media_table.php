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
        Schema::create('lesson_media', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('lesson_id');
            $table->enum('type', ['video', 'audio', 'image', 'sign3d', 'document'])->default('video');
            $table->string('provider', 50)->default('cdn'); // cdn, youtube, vimeo, etc.
            $table->string('storage_path', 500)->nullable();
            $table->string('url', 500)->nullable();
            $table->string('mime', 100)->nullable();
            $table->integer('duration')->nullable(); // секунды
            $table->string('quality', 20)->nullable(); // 720p, 1080p, master, etc.
            $table->json('sources')->nullable(); // массив источников
            $table->json('captions')->nullable(); // субтитры
            $table->string('poster_url', 500)->nullable();
            $table->boolean('is_default')->default(false);
            $table->json('extra')->nullable(); // доп. данные
            $table->timestamps();
            
            $table->foreign('lesson_id')->references('id')->on('lessons')->onDelete('cascade');
            $table->index(['lesson_id', 'type']);
            $table->index(['lesson_id', 'is_default']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('lesson_media');
    }
};

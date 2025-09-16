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
        Schema::create('lesson_texts', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('lesson_id');
            $table->string('language', 10)->default('ru'); // ru, kz, asl, etc.
            $table->enum('format', ['markdown', 'html', 'plain'])->default('markdown');
            $table->longText('content');
            $table->integer('reading_time')->nullable(); // минуты
            $table->boolean('is_primary')->default(false);
            $table->json('meta')->nullable(); // доп. метаданные
            $table->timestamps();
            
            $table->foreign('lesson_id')->references('id')->on('lessons')->onDelete('cascade');
            $table->index(['lesson_id', 'language']);
            $table->index(['lesson_id', 'is_primary']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('lesson_texts');
    }
};

"""
dev/train_router.py

Extract training data from conversation history and train the RouteLLM router.

Training data:
- Prompts where local model succeeded (quality_passed=True) → label 0 (local)
- Prompts where local failed and escalated to cloud → label 1 (cloud needed)
- Prompts routed directly to cloud (complexity=high) → label 1 (cloud needed)
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import select, and_
from db.engine import init_db, close_db, get_db_session
from db.models import ConversationMessage
from src.router import RouteLLMRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def extract_training_data():
    """
    Extract training data from conversation_messages table.
    
    Returns:
        Tuple of (prompts, labels) where:
        - prompts: List of user query strings
        - labels: List of 0 (local sufficient) or 1 (cloud needed)
    """
    logger.info("Extracting training data from database...")
    
    await init_db()
    
    prompts = []
    labels = []
    
    async with get_db_session() as session:
        # Get all user messages with their subsequent assistant responses
        result = await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.role == "user")
            .order_by(ConversationMessage.created_at)
        )
        user_messages = result.scalars().all()
        
        logger.info(f"Found {len(user_messages)} user messages")
        
        for user_msg in user_messages:
            # Find the next assistant message
            assistant_result = await session.execute(
                select(ConversationMessage)
                .where(
                    and_(
                        ConversationMessage.conversation_id == user_msg.conversation_id,
                        ConversationMessage.role == "assistant",
                        ConversationMessage.created_at > user_msg.created_at,
                    )
                )
                .order_by(ConversationMessage.created_at)
                .limit(1)
            )
            assistant_msg = assistant_result.scalars().first()
            
            if not assistant_msg:
                continue
            
            prompt = user_msg.content
            backend = assistant_msg.backend
            quality_passed = assistant_msg.quality_passed
            
            # Determine label
            # 0 = local was sufficient
            # 1 = cloud was needed
            if backend == "local" and quality_passed:
                # Local succeeded
                label = 0
            elif backend == "fireworks":
                # Cloud was used (either directly routed or escalated)
                label = 1
            elif backend == "local" and quality_passed is False:
                # Local failed quality check, should have used cloud
                label = 1
            else:
                # Unclear, skip
                continue
            
            prompts.append(prompt)
            labels.append(label)
            
            logger.debug(f"Added: '{prompt[:50]}...' → {'local' if label == 0 else 'cloud'}")
    
    await close_db()
    
    logger.info(f"Extracted {len(prompts)} training examples")
    logger.info(f"  - Local successful: {labels.count(0)}")
    logger.info(f"  - Cloud needed: {labels.count(1)}")
    
    return prompts, labels


def add_synthetic_data(prompts: list[str], labels: list[int]) -> tuple[list[str], list[int]]:
    """
    Add synthetic training data for common patterns.
    
    This helps when we don't have enough real data yet.
    """
    logger.info("Adding synthetic training examples...")
    

    # REBALANCED: More educational queries labeled as local-appropriate
    synthetic_local = [
        # Math questions
        "What is 2 + 2?",
        "Calculate 15 * 7", 
        "Solve: 100 / 4",
        "What's 25% of 80?",
        
        # Educational/definitional queries (MOVED FROM CLOUD)
        "What is Python?",
        "Define machine learning",
        "What does API stand for?",
        "What is HTTP?",
        "Define JSON",
        "What is Git?",
        "Define CPU",
        "What is RAM?",
        "Explain what tuples are in Python",
        "What is object-oriented programming?",
        "Define recursion in programming",
        "What are variables in programming?",
        "Explain the basics of databases",
        "What is a function in programming?",
        "Define algorithm",
        
        # Simple factual
        "Who invented the internet?",
        "How many days in a week?",
        "What color is the sky?",
        "When was Python created?",
        
        # Simple greetings/responses  
        "Hello",
        "Thank you",
    ]
    
    synthetic_cloud = [
        # Complex reasoning (KEPT IN CLOUD)
        "Explain the philosophical implications of artificial intelligence consciousness",
        "Analyze the economic factors that led to the 2008 financial crisis", 
        "Compare and contrast quantum mechanics with classical physics",
        "Elaborate on the socio-political ramifications of climate change",
        
        # Complex analysis requiring depth
        "Analyze the impact of social media on mental health with supporting research",
        "Discuss the ethical implications of gene editing in detail", 
        "Provide a comprehensive comparison of different economic systems",
        "Evaluate the long-term effects of remote work on corporate culture",
        
        # Long-form content creation
        "Write a comprehensive guide to machine learning algorithms with examples",
        "Create a detailed business proposal for a renewable energy startup",
        "Draft a marketing strategy for launching a new tech product",
        "Write a research paper on climate change impacts",
        
        # Complex technical implementation
        "Debug this complex Python code and explain the issue in detail",
        "Architect a scalable microservices system for e-commerce with security considerations",
        "Design a distributed database system for high availability",
        "Implement a machine learning pipeline for real-time data processing",
        
        # Creative writing
        "Write a short story about time travel with plot twists",
        "Create a poem about the ocean with specific literary devices", 
        "Draft a screenplay for a sci-fi movie",
        
        # Multi-step reasoning
        "How would you solve world hunger? Provide a detailed multi-phase plan",
        "Explain the causes and effects of the Industrial Revolution with timeline",
        "Design a complete urban planning strategy for sustainable cities",
        
        # Deep technical explanations requiring context
        "Explain how modern neural networks learn from data with mathematical foundations",
        "Describe the complete architecture of a modern operating system with examples",
        "Analyze the trade-offs between different distributed system architectures",
    ]
    
    # Add synthetic data
    for q in synthetic_local:
        prompts.append(q)
        labels.append(0)  # Local
    
    for q in synthetic_cloud:
        prompts.append(q)
        labels.append(1)  # Cloud
    
    logger.info(f"Added {len(synthetic_local)} synthetic local examples")
    logger.info(f"Added {len(synthetic_cloud)} synthetic cloud examples")
    
    return prompts, labels


async def main():
    """Main training script."""
    logger.info("=" * 60)
    logger.info("RouteLLM Router Training")
    logger.info("=" * 60)
    
    # Extract real data
    prompts, labels = await extract_training_data()
    
    # Add synthetic data if we don't have enough balanced real data
    if len(prompts) < 50 or labels.count(0) < 15 or labels.count(1) < 15:
        logger.warning(
            f"Data is sparse or imbalanced ({len(prompts)} examples: {labels.count(0)} local, {labels.count(1)} cloud). "
            f"Supplementing with synthetic training examples..."
        )
        prompts, labels = add_synthetic_data(prompts, labels)
    
    if len(prompts) < 5:
        logger.error("Not enough training data. Need at least 5 examples.")
        return
    
    logger.info(f"Total training data: {len(prompts)} examples")
    logger.info(f"  - Local: {labels.count(0)} ({labels.count(0)/len(labels)*100:.1f}%)")
    logger.info(f"  - Cloud: {labels.count(1)} ({labels.count(1)/len(labels)*100:.1f}%)")
    
    # Train router with calibrated threshold
    logger.info("\nInitializing router...")
    router = RouteLLMRouter(
        embedding_model="all-MiniLM-L6-v2",
        threshold=0.15,
        enable_fallback=True,
    )
    
    logger.info("Training router (this may take 30-60 seconds)...")
    router.train(
        prompts=prompts,
        labels=labels,
        learning_rate=0.01,
        regularization=0.1,
    )
    
    # Save model
    model_path = "models/router.pkl"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    router.save(model_path)
    
    logger.info(f"\n✓ Router trained and saved to {model_path}")
    
    # Test predictions
    logger.info("\n" + "=" * 60)
    logger.info("Testing Router Predictions")
    logger.info("=" * 60)
    
    test_prompts = [
        "What is 2+2?",
        "Explain quantum mechanics in detail",
        "Define Python",
        "Analyze the economic implications of AI",
    ]
    
    for test_prompt in test_prompts:
        decision = router.predict_route(test_prompt, context_length=0)
        logger.info(f"\nPrompt: '{test_prompt}'")
        logger.info(f"  → {decision.backend.upper()}")
        logger.info(f"  → Score: {decision.score:.3f}")
        logger.info(f"  → Confidence: {decision.confidence:.2f}")
        logger.info(f"  → Complexity: {decision.complexity}")
        logger.info(f"  → Reason: {decision.reason}")
    
    # Print stats
    logger.info("\n" + "=" * 60)
    logger.info("Router Statistics")
    logger.info("=" * 60)
    stats = router.get_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\n✓ Training complete!")


if __name__ == "__main__":
    asyncio.run(main())

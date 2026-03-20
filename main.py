"""
Main pipeline execution script for Deep Learning pipeline.

Author: Grzegorz Gomza
Date: February 2026
"""

from textwrap import dedent

from src.WasteClassifier import logger
from src.WasteClassifier.config.configuration import ConfigurationManager

# DL Pipeline imports
from src.WasteClassifier.pipeline.stage_01_data_ingestion import DLDataIngestionPipeline
from src.WasteClassifier.pipeline.stage_02_prepare_models import DLPrepareBaseModelPipeline
from src.WasteClassifier.pipeline.stage_03_train import DLTrainModelPipeline
from src.WasteClassifier.pipeline.stage_04_evaluate import DLEvaluatePipeline

# ML Pipeline imports removed - focusing on DL only

class RunDLPipeline:
    # Deep Learning Pipeline Stages
    @staticmethod
    def stage_01_data_ingestion():
        STAGE_NAME = "DL Data Ingestion Stage"
        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DLDataIngestionPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

    @staticmethod
    def stage_02_prepare_models():
        STAGE_NAME = "DL Prepare Base Model Stage"
        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DLPrepareBaseModelPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

    @staticmethod
    def stage_03_train():
        STAGE_NAME = "DL Training Stage"
        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DLTrainModelPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

    @staticmethod
    def stage_04_evaluate():
        STAGE_NAME = "DL Evaluation Stage"
        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DLEvaluatePipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

# ML Pipeline class removed - focusing on DL only

def run_dl_pipeline(num_stages):
    """Run Deep Learning Pipeline."""
    dl_stages = [
        RunDLPipeline.stage_01_data_ingestion,
        RunDLPipeline.stage_02_prepare_models,
        RunDLPipeline.stage_03_train,
        RunDLPipeline.stage_04_evaluate,
    ]

    logger.info("🚀 Starting DEEP LEARNING Pipeline")
    logger.info("="*50)

    for i in range(min(num_stages, len(dl_stages))):
        stage_name = dl_stages[i].__name__
        logger.info(f">>>>>> DL Stage {i+1}: {stage_name} <<<<<")
        dl_stages[i]()

    logger.info("🎉 Deep Learning Pipeline completed!")

# ML pipeline function removed - focusing on DL only

def main():
    """
    Main function with simplified pipeline selection.
    """
    print(dedent("""
        ============================================================
        🧠 WASTE CLASSIFICATION PIPELINE SYSTEM
        ============================================================
        
        � Deep Learning Pipeline (Transfer Learning)
        - Uses pre-trained models (ResNet50, MobileNetV2, EfficientNet-B0)
        - Fine-tunes on waste dataset
        - Compares 3 different base models for performance
        
        � Pipeline Stages:
        1. Data Ingestion - Download and prepare the waste dataset
        2. Prepare Base Models - Load and customize pre-trained models
        3. Train DL Models - Train all 3 models on the waste dataset
        4. Evaluate DL Models - Compare performance and generate reports
    """))

    # Single user interaction - stage selection
    max_stages = 4
    stage_descriptions = {
        1: "Data Ingestion only",
        2: "Data Ingestion + Model Preparation",
        3: "Data Ingestion + Model Prep + Training",
        4: "Complete Pipeline (All stages including evaluation)"
    }
    
    print("\n" + "="*60)
    print("📊 Select how many stages to run:")
    for i in range(1, max_stages + 1):
        print(f"   {i}. {stage_descriptions[i]}")
    
    while True:
        try:
            num_stages = input(f"\n🎯 Enter stage number (1-{max_stages}): ").strip()
            num_stages = int(num_stages)
            if 1 <= num_stages <= max_stages:
                print(f"\n✅ Running: {stage_descriptions[num_stages]}")
                break
            else:
                print(f"❌ Invalid number. Please enter between 1 and {max_stages}.")
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            return
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Run DL pipeline
    try:
        print("\n" + "="*60)
        print(f"🚀 Starting Deep Learning Pipeline - Stages 1-{num_stages}")
        print("="*60)
        
        logger.info(f">>>>> Deep Learning pipeline started (stages: 1-{num_stages}) <<<<<")
        logger.info(">>>>>> Running Deep Learning pipeline with 3 base models <<<<<")
        run_dl_pipeline(num_stages)
        logger.info(">>>>>> Deep Learning pipeline completed <<<<<")
        
        print("\n" + "="*60)
        print(f"🎉 Pipeline completed successfully!")
        print(f"📋 Completed: {stage_descriptions[num_stages]}")
        print(f"🤖 Models used: ResNet50, MobileNetV2, EfficientNet-B0")
        print("="*60)
        
        logger.info(f">>>>> Pipeline completed successfully for stages 1-{num_stages} <<<<<")
        
    except Exception as e:
        logger.exception(e)
        logger.error(">>>>> Pipeline failed <<<<<")
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("🔧 Please check the logs above for details")
        raise e

if __name__ == "__main__":
    main()
    print("\n✅ Pipeline execution finished!")
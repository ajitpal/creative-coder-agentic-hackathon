#!/usr/bin/env python3
"""
Command Line Interface for Intelligent Healthcare Navigator
Provides text-based interaction with the healthcare navigation system
"""

import asyncio
import argparse
import sys
import os
from typing import Dict, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agent import HealthcareNavigatorAgent
from src.config import Config
from src.utils import setup_logging

logger = setup_logging()

class HealthcareCLI:
    """Command line interface for healthcare navigation"""
    
    def __init__(self):
        """Initialize CLI"""
        self.agent = None
        self.session_id = "cli_session"
    
    async def initialize(self):
        """Initialize the healthcare agent"""
        try:
            self.agent = HealthcareNavigatorAgent(self.session_id)
            print("🏥 Healthcare Navigator CLI initialized successfully!")
            print("Type 'help' for available commands or 'quit' to exit.\n")
        except Exception as e:
            print(f"❌ Failed to initialize Healthcare Navigator: {e}")
            sys.exit(1)
    
    def print_help(self):
        """Print help information"""
        help_text = """
🏥 Healthcare Navigator CLI Commands:

Basic Commands:
  help                    - Show this help message
  quit, exit             - Exit the application
  clear                  - Clear conversation history
  status                 - Show system status

Query Commands:
  ask <question>         - Ask a medical question
  drug <drug_name>       - Get drug information
  symptom <symptoms>     - Analyze symptoms
  term <medical_term>    - Look up medical term

Document Commands:
  upload <file_path>     - Upload and analyze a medical document
  
Preference Commands:
  set <key> <value>      - Set user preference
  get <key>              - Get user preference
  
Examples:
  ask "What is diabetes?"
  drug "aspirin"
  symptom "headache and fever"
  term "hypertension"
  upload "medical_report.pdf"
  set age 35
        """
        print(help_text)
    
    async def process_command(self, command: str) -> bool:
        """Process user command. Returns False if should quit."""
        command = command.strip()
        
        if not command:
            return True
        
        # Parse command
        parts = command.split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd in ['quit', 'exit']:
                return False
            
            elif cmd == 'help':
                self.print_help()
            
            elif cmd == 'clear':
                success = self.agent.clear_conversation_history()
                if success:
                    print("✅ Conversation history cleared.")
                else:
                    print("❌ Failed to clear conversation history.")
            
            elif cmd == 'status':
                await self.show_status()
            
            elif cmd == 'ask':
                if not args:
                    print("❌ Please provide a question. Example: ask 'What is diabetes?'")
                else:
                    await self.process_query(args)
            
            elif cmd == 'drug':
                if not args:
                    print("❌ Please provide a drug name. Example: drug aspirin")
                else:
                    await self.process_query(f"Tell me about the drug {args}")
            
            elif cmd == 'symptom':
                if not args:
                    print("❌ Please provide symptoms. Example: symptom 'headache and fever'")
                else:
                    await self.process_query(f"I have these symptoms: {args}")
            
            elif cmd == 'term':
                if not args:
                    print("❌ Please provide a medical term. Example: term hypertension")
                else:
                    await self.process_query(f"What is {args}?")
            
            elif cmd == 'upload':
                if not args:
                    print("❌ Please provide a file path. Example: upload medical_report.pdf")
                else:
                    await self.upload_document(args)
            
            elif cmd == 'set':
                await self.set_preference(args)
            
            elif cmd == 'get':
                await self.get_preference(args)
            
            else:
                print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
        
        except Exception as e:
            print(f"❌ Error processing command: {e}")
            logger.error(f"CLI command error: {e}")
        
        return True
    
    async def process_query(self, query: str):
        """Process a medical query"""
        print(f"\n🤔 Processing: {query}")
        print("⏳ Please wait...")
        
        try:
            response = await self.agent.process_query(query)
            
            print(f"\n📋 Response:")
            print("=" * 60)
            print(response.response.response_text)
            
            if response.response.sources:
                print(f"\n📚 Sources: {', '.join(response.response.sources)}")
            
            if response.response.confidence_score:
                print(f"🎯 Confidence: {response.response.confidence_score:.1%}")
            
            if response.response.disclaimers:
                print(f"\n⚠️ Important Disclaimers:")
                for disclaimer in response.response.disclaimers:
                    print(f"• {disclaimer}")
            
            print(f"\n⏱️ Processing time: {response.processing_time:.2f}s")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    async def upload_document(self, file_path: str):
        """Upload and process a document"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return
            
            print(f"\n📄 Uploading document: {file_path}")
            print("⏳ Processing...")
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            file_size = len(file_data)
            filename = os.path.basename(file_path)
            
            response = await self.agent.handle_document_upload(file_data, filename, file_size)
            
            print(f"\n📋 Document Analysis:")
            print("=" * 60)
            print(response.response.response_text)
            
            if response.response.disclaimers:
                print(f"\n⚠️ Important Notes:")
                for disclaimer in response.response.disclaimers:
                    print(f"• {disclaimer}")
            
            print(f"\n⏱️ Processing time: {response.processing_time:.2f}s")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Document upload failed: {e}")
    
    async def show_status(self):
        """Show system status"""
        try:
            status = await self.agent.get_system_status()
            
            print(f"\n🔍 System Status:")
            print("=" * 40)
            print(f"Session ID: {status['session_id']}")
            print(f"System Healthy: {'✅' if status['system_healthy'] else '❌'}")
            
            print(f"\n📊 Components:")
            for component, healthy in status['component_status'].items():
                status_icon = '✅' if healthy else '❌'
                print(f"  {status_icon} {component}")
            
            if 'session_stats' in status:
                stats = status['session_stats']
                print(f"\n💬 Session Stats:")
                print(f"  Interactions: {stats.get('total_interactions', 0)}")
                print(f"  Last activity: {stats.get('last_activity', 'Never')}")
            
            print("=" * 40)
            
        except Exception as e:
            print(f"❌ Status check failed: {e}")
    
    async def set_preference(self, args: str):
        """Set user preference"""
        try:
            parts = args.split(' ', 1)
            if len(parts) != 2:
                print("❌ Usage: set <key> <value>")
                return
            
            key, value = parts
            success = self.agent.set_user_preference(key, value)
            
            if success:
                print(f"✅ Preference set: {key} = {value}")
            else:
                print(f"❌ Failed to set preference: {key}")
                
        except Exception as e:
            print(f"❌ Error setting preference: {e}")
    
    async def get_preference(self, key: str):
        """Get user preference"""
        try:
            if not key:
                print("❌ Usage: get <key>")
                return
            
            value = self.agent.get_user_preference(key)
            
            if value is not None:
                print(f"📋 {key} = {value}")
            else:
                print(f"❌ Preference not found: {key}")
                
        except Exception as e:
            print(f"❌ Error getting preference: {e}")
    
    async def run_interactive(self):
        """Run interactive CLI mode"""
        await self.initialize()
        
        print("💬 Interactive mode started. Type your commands below:")
        
        try:
            while True:
                try:
                    command = input("\n🏥 > ").strip()
                    
                    if not await self.process_command(command):
                        break
                        
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break
                    
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            logger.error(f"CLI runtime error: {e}")

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Intelligent Healthcare Navigator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py                           # Interactive mode
  python cli.py --query "What is diabetes?"
  python cli.py --drug "aspirin"
  python cli.py --upload "report.pdf"
        """
    )
    
    parser.add_argument('--query', '-q', help='Ask a medical question')
    parser.add_argument('--drug', '-d', help='Get drug information')
    parser.add_argument('--symptom', '-s', help='Analyze symptoms')
    parser.add_argument('--term', '-t', help='Look up medical term')
    parser.add_argument('--upload', '-u', help='Upload and analyze document')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--session', default='cli_session', help='Session ID')
    
    args = parser.parse_args()
    
    cli = HealthcareCLI()
    cli.session_id = args.session
    
    # Handle single commands
    if args.status:
        await cli.initialize()
        await cli.show_status()
        return
    
    if args.query:
        await cli.initialize()
        await cli.process_query(args.query)
        return
    
    if args.drug:
        await cli.initialize()
        await cli.process_query(f"Tell me about the drug {args.drug}")
        return
    
    if args.symptom:
        await cli.initialize()
        await cli.process_query(f"I have these symptoms: {args.symptom}")
        return
    
    if args.term:
        await cli.initialize()
        await cli.process_query(f"What is {args.term}?")
        return
    
    if args.upload:
        await cli.initialize()
        await cli.upload_document(args.upload)
        return
    
    # Default to interactive mode
    await cli.run_interactive()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)
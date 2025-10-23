#!/usr/bin/env python3
"""
Simple test script to verify OpenTelemetry instrumentation is working.
This script tests the basic OpenTelemetry setup without running the full Chainlit app.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../variables.env")

def test_otel_imports():
    """Test that all OpenTelemetry imports work correctly"""
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        print("✅ All OpenTelemetry imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment_variables():
    """Test that required environment variables are set"""
    required_vars = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS", 
        "OTEL_RESOURCE_ATTRIBUTES"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def test_otel_initialization():
    """Test OpenTelemetry initialization"""
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        
        # Create resource with attributes
        resource = Resource.create({
            "service.name": "home-finder-test",
            "service.version": "1.0",
            "deployment.environment": "test"
        })
        
        # Configure OTLP exporters
        otlp_endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        otlp_headers = os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
        
        # Parse headers - handle both comma-separated and single header formats
        # For gRPC, we need to ensure headers are in the correct format
        headers = {}
        if "," in otlp_headers:
            # Multiple headers separated by commas
            for header in otlp_headers.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    key = key.strip().lower()  # gRPC requires lowercase keys
                    value = value.strip()
                    headers[key] = value
        else:
            # Single header
            if "=" in otlp_headers:
                key, value = otlp_headers.split("=", 1)
                key = key.strip().lower()  # gRPC requires lowercase keys
                value = value.strip()
                headers[key] = value
        
        # Initialize span exporter
        span_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            headers=headers
        )
        
        # Initialize metric exporter
        metric_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint,
            headers=headers
        )
        
        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        span_processor = BatchSpanProcessor(span_exporter)
        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)
        
        # Set up meter provider
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=5000  # Export every 5 seconds for testing
        )
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)
        
        print("✅ OpenTelemetry initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ OpenTelemetry initialization failed: {e}")
        return False

def test_tracing():
    """Test basic tracing functionality"""
    try:
        from opentelemetry import trace
        
        tracer = trace.get_tracer(__name__)
        
        with tracer.start_as_current_span("test_span") as span:
            span.set_attribute("test.attribute", "test_value")
            span.add_event("test_event", {"key": "value"})
            time.sleep(0.1)  # Simulate some work
        
        print("✅ Basic tracing test successful")
        return True
        
    except Exception as e:
        print(f"❌ Tracing test failed: {e}")
        return False

def test_metrics():
    """Test basic metrics functionality"""
    try:
        from opentelemetry import metrics
        
        meter = metrics.get_meter(__name__)
        
        # Create a counter
        counter = meter.create_counter(
            name="test.counter",
            description="A test counter"
        )
        
        # Create a histogram
        histogram = meter.create_histogram(
            name="test.histogram",
            description="A test histogram"
        )
        
        # Record some metrics
        counter.add(1, {"test": "value"})
        histogram.record(100, {"test": "value"})
        
        print("✅ Basic metrics test successful")
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing OpenTelemetry instrumentation...")
    print("=" * 50)
    
    tests = [
        ("OpenTelemetry Imports", test_otel_imports),
        ("Environment Variables", test_environment_variables),
        ("OpenTelemetry Initialization", test_otel_initialization),
        ("Basic Tracing", test_tracing),
        ("Basic Metrics", test_metrics),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed: {test_name}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! OpenTelemetry instrumentation is working correctly.")
        print("\n📝 Next steps:")
        print("1. Run the Chainlit app: chainlit run app.py --port 8022")
        print("2. Send some messages to generate telemetry data")
        print("3. Check your Elasticsearch instance for traces and metrics")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

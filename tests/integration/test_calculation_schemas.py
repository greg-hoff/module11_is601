# tests/integration/test_calculation_schemas.py
"""
Integration Tests for Calculation Pydantic Schemas

These tests verify that Pydantic schemas correctly validate calculation data
before it reaches the application logic. This is an important security and
data integrity layer that prevents invalid data from entering the system.

Key Testing Concepts:
1. Valid Data: Ensure schemas accept correct data
2. Invalid Data: Ensure schemas reject incorrect data with clear messages
3. Edge Cases: Test boundary conditions
4. Business Rules: Verify domain-specific validation (e.g., no division by 0)
"""

import pytest
import uuid
from datetime import datetime
from pydantic import ValidationError
from typing import List

from app.schemas.calculation import (
    CalculationType,
    CalculationBase,
    CalculationCreate,
    CalculationUpdate,
    CalculationResponse
)


@pytest.fixture
def sample_user_id():
    """Fixture providing a consistent user ID for tests."""
    return uuid.uuid4()


@pytest.fixture
def sample_datetime():
    """Fixture providing a consistent datetime for tests."""
    return datetime(2025, 1, 1, 12, 0, 0)


def assert_validation_error_contains(exc_info, expected_message: str, field_name: str = None):
    """Helper function to check validation error content."""
    errors = exc_info.value.errors()
    assert len(errors) >= 1
    
    if field_name:
        field_errors = [err for err in errors if field_name in str(err["loc"])]
        assert len(field_errors) >= 1
        assert any(expected_message in err["msg"] for err in field_errors)
    else:
        assert any(expected_message in err["msg"] for err in errors)


class TestCalculationType:
    """Test the CalculationType enum functionality."""
    
    def test_enum_properties(self):
        """Test enum values, string inheritance, and iteration."""
        # Test expected values
        expected_types = {"addition", "subtraction", "multiplication", "division"}
        actual_types = {calc_type.value for calc_type in CalculationType}
        assert actual_types == expected_types
        
        # Test string inheritance for JSON serialization
        assert isinstance(CalculationType.ADDITION, str)
        assert CalculationType.ADDITION == "addition"
        
        # Test iteration
        types = list(CalculationType)
        assert len(types) == 4
        assert all(isinstance(calc_type, CalculationType) for calc_type in types)


class TestCalculationBase:
    """Test the base calculation schema validation."""
    
    @pytest.mark.parametrize("calc_type,inputs", [
        ("addition", [1, 2]),
        ("subtraction", [10, 5]), 
        ("multiplication", [3, 4]),
        ("division", [100, 2])
    ])
    def test_valid_calculation_data_individual_operations(self, calc_type: str, inputs: List[float]):
        """Test that each calculation type accepts valid data individually."""
        data = {"type": calc_type, "inputs": inputs}
        calc = CalculationBase(**data)
        
        assert calc.type == calc_type
        assert calc.inputs == inputs
    
    def test_case_insensitive_and_invalid_types(self):
        """Test type validation including case-insensitivity and invalid types."""
        # Test case-insensitive validation
        for calc_type in ["ADDITION", "Addition", "MULTIPLICATION", "Division"]:
            data = {"type": calc_type, "inputs": [1, 2]}
            calc = CalculationBase(**data)
            assert calc.type == calc_type.lower()
        
        # Test invalid types
        invalid_types = ["invalid", "add", "power", "", None, 123]
        for invalid_type in invalid_types:
            data = {"type": invalid_type, "inputs": [1, 2]}
            with pytest.raises(ValidationError) as exc_info:
                CalculationBase(**data)
            assert_validation_error_contains(exc_info, "Type must be one of:", "type")
    
    def test_inputs_validation_comprehensive(self):
        """Test comprehensive input validation including format and values."""
        # Test invalid input formats
        invalid_formats = [[], [1], "not a list", 123, None, {"a": 1}]
        for invalid_inputs in invalid_formats:
            data = {"type": "addition", "inputs": invalid_inputs}
            with pytest.raises(ValidationError):
                CalculationBase(**data)
        
        # Test invalid input values
        invalid_values = [[1, "invalid"], [1, None, 3], [1, [], 3]]
        for inputs in invalid_values:
            data = {"type": "addition", "inputs": inputs}
            with pytest.raises(ValidationError):
                CalculationBase(**data)
    
    def test_division_by_zero_comprehensive(self):
        """Test division by zero prevention comprehensively."""
        # Single zero denominator
        data = {"type": "division", "inputs": [10, 0]}
        with pytest.raises(ValidationError) as exc_info:
            CalculationBase(**data)
        assert_validation_error_contains(exc_info, "Cannot divide by zero")
        
        # Zero in multiple denominators
        data = {"type": "division", "inputs": [100, 2, 0, 5]}
        with pytest.raises(ValidationError) as exc_info:
            CalculationBase(**data)
        assert_validation_error_contains(exc_info, "Cannot divide by zero")
        
        # Allow zero numerator
        data = {"type": "division", "inputs": [0, 5]}
        calc = CalculationBase(**data)
        assert calc.type == "division"
        assert calc.inputs == [0, 5]
        
        # Allow zeros in non-division operations
        for calc_type in ["addition", "subtraction", "multiplication"]:
            data = {"type": calc_type, "inputs": [0, 0, 5]}
            calc = CalculationBase(**data)
            assert calc.type == calc_type
    
    def test_special_numeric_values_and_edge_cases(self):
        """Test special numeric values and edge cases."""
        special_cases = [
            [1.5, 2.7],              # Decimals
            [1e10, 2e10],            # Large numbers
            [1e-10, 2e-10],          # Small numbers
            [-1.5, 2.7],             # Negative numbers
            [float('inf'), 1],       # Infinity
            [-float('inf'), 1],      # Negative infinity
            list(range(1, 101))      # Large input list (100 numbers)
        ]
        
        for inputs in special_cases:
            data = {"type": "addition", "inputs": inputs}
            calc = CalculationBase(**data)
            assert calc.type == "addition"
            assert calc.inputs == inputs
    
    def test_nan_values_handling(self):
        """Test that NaN values are handled properly."""
        data = {"type": "addition", "inputs": [1, float('nan')]}
        calc = CalculationBase(**data)
        
        assert calc.type == "addition"
        assert len(calc.inputs) == 2
        assert calc.inputs[0] == 1
        # Note: NaN != NaN, so we don't directly test the second value


class TestCalculationCreate:
    """Test the CalculationCreate schema validation."""
    
    @pytest.mark.parametrize("calc_type,inputs", [
        ("addition", [1.5, 2.5]),
        ("subtraction", [10, 5]),
        ("multiplication", [3, 4]),
        ("division", [100, 2])
    ])
    def test_valid_creation_individual_operations(self, calc_type: str, inputs: List[float], sample_user_id):
        """Test that each calculation type can be created individually."""
        data = {
            "type": calc_type,
            "inputs": inputs,
            "user_id": str(sample_user_id)
        }
        
        calc = CalculationCreate(**data)
        
        assert calc.type == calc_type
        assert calc.inputs == inputs
        assert calc.user_id == sample_user_id
    
    def test_user_id_validation_comprehensive(self, sample_user_id):
        """Test user_id validation including conversion and invalid values."""
        # Test string to UUID conversion
        user_id_str = "123e4567-e89b-12d3-a456-426614174000"
        data = {
            "type": "multiplication",
            "inputs": [3, 4],
            "user_id": user_id_str
        }
        calc = CalculationCreate(**data)
        assert isinstance(calc.user_id, uuid.UUID)
        assert str(calc.user_id) == user_id_str
        
        # Test invalid user_ids
        invalid_user_ids = ["not-a-uuid", "123", "", None, 123, []]
        for invalid_user_id in invalid_user_ids:
            data = {
                "type": "addition",
                "inputs": [1, 2],
                "user_id": invalid_user_id
            }
            with pytest.raises(ValidationError) as exc_info:
                CalculationCreate(**data)
            assert_validation_error_contains(exc_info, "", "user_id")
    
    def test_required_fields_and_inheritance(self):
        """Test required fields validation and inheritance from base."""
        # Test missing required fields
        with pytest.raises(ValidationError) as exc_info:
            CalculationCreate(type="addition", inputs=[1, 2])  # Missing user_id
        missing_fields = [err["loc"][0] for err in exc_info.value.errors() if err["type"] == "missing"]
        assert "user_id" in missing_fields
        
        # Test inheritance from CalculationBase (division by zero)
        user_id = uuid.uuid4()
        with pytest.raises(ValidationError) as exc_info:
            CalculationCreate(type="division", inputs=[10, 0], user_id=str(user_id))
        assert_validation_error_contains(exc_info, "Cannot divide by zero")


class TestCalculationUpdate:
    """Test the CalculationUpdate schema validation."""
    
    def test_update_validation_comprehensive(self):
        """Test comprehensive update validation including optional fields and constraints."""
        # Test valid update
        calc = CalculationUpdate(inputs=[10.5, 5.5])
        assert calc.inputs == [10.5, 5.5]
        
        # Test optional inputs (no data provided)
        calc = CalculationUpdate()
        assert calc.inputs is None
        
        calc = CalculationUpdate(**{})
        assert calc.inputs is None
        
        # Test input validation when provided
        with pytest.raises(ValidationError) as exc_info:
            CalculationUpdate(inputs=[1])  # Too few inputs
        # The error message is from Pydantic's min_length validation, not our custom message
        assert_validation_error_contains(exc_info, "at least 2 items", "inputs")
        
        # Test invalid input types
        with pytest.raises(ValidationError):
            CalculationUpdate(inputs=[1, "invalid"])
        
        # Test large input list
        large_inputs = list(range(1, 21))
        calc = CalculationUpdate(inputs=large_inputs)
        assert calc.inputs == large_inputs


class TestCalculationResponse:
    """Test the CalculationResponse schema validation."""
    
    @pytest.mark.parametrize("calc_type,inputs,result", [
        ("addition", [10.5, 5.5], 16.0),
        ("subtraction", [10, 5], 5.0),
        ("multiplication", [3, 4], 12.0),
        ("division", [100, 4], 25.0)
    ])
    def test_valid_response_individual_operations(self, calc_type: str, inputs: List[float], result: float, sample_user_id, sample_datetime):
        """Test that each calculation type produces valid response individually."""
        calc_id = uuid.uuid4()
        data = {
            "id": str(calc_id),
            "user_id": str(sample_user_id),
            "type": calc_type,
            "inputs": inputs,
            "result": result,
            "created_at": sample_datetime.isoformat(),
            "updated_at": sample_datetime.isoformat()
        }
        
        calc = CalculationResponse(**data)
        
        assert calc.id == calc_id
        assert calc.user_id == sample_user_id
        assert calc.type == calc_type
        assert calc.inputs == inputs
        assert calc.result == result
        assert isinstance(calc.created_at, datetime)
        assert isinstance(calc.updated_at, datetime)
    
    def test_datetime_and_field_validation_comprehensive(self, sample_user_id):
        """Test comprehensive datetime parsing and field validation."""
        calc_id = uuid.uuid4()
        
        # Test various datetime formats
        datetime_formats = [
            "2025-01-01T12:00:00",           # Basic format
            "2025-01-01T12:00:00Z",          # UTC timezone
            "2025-01-01T12:30:00+05:00"      # Offset timezone
        ]
        
        for dt_format in datetime_formats:
            data = {
                "id": str(calc_id),
                "user_id": str(sample_user_id),
                "type": "multiplication",
                "inputs": [3, 4],
                "result": 12.0,
                "created_at": dt_format,
                "updated_at": dt_format
            }
            calc = CalculationResponse(**data)
            assert isinstance(calc.created_at, datetime)
            assert isinstance(calc.updated_at, datetime)
        
        # Test different result numeric types
        result_types = [3, 3.14, "3.14"]  # int, float, string number
        for result_val in result_types:
            data = {
                "id": str(calc_id),
                "user_id": str(sample_user_id),
                "type": "addition",
                "inputs": [1, 2],
                "result": result_val,
                "created_at": "2025-01-01T12:00:00",
                "updated_at": "2025-01-01T12:00:00"
            }
            calc = CalculationResponse(**data)
            assert isinstance(calc.result, float)
    
    def test_required_fields_and_inheritance(self, sample_user_id, sample_datetime):
        """Test required fields validation and inheritance from base."""
        calc_id = uuid.uuid4()
        complete_data = {
            "id": str(calc_id),
            "user_id": str(sample_user_id),
            "type": "addition",
            "inputs": [1, 2],
            "result": 3.0,
            "created_at": sample_datetime.isoformat(),
            "updated_at": sample_datetime.isoformat()
        }
        
        # Test missing each required field individually
        required_fields = ["id", "user_id", "type", "inputs", "result", "created_at", "updated_at"]
        for missing_field in required_fields:
            incomplete_data = {k: v for k, v in complete_data.items() if k != missing_field}
            with pytest.raises(ValidationError) as exc_info:
                CalculationResponse(**incomplete_data)
            missing_fields = [err["loc"][0] for err in exc_info.value.errors() if err["type"] == "missing"]
            assert missing_field in missing_fields
        
        # Test inheritance from CalculationBase (invalid type)
        with pytest.raises(ValidationError):
            CalculationResponse(
                id=str(calc_id),
                user_id=str(sample_user_id),
                type="invalid_type",
                inputs=[1, 2],
                result=3.0,
                created_at=sample_datetime.isoformat(),
                updated_at=sample_datetime.isoformat()
            )


class TestSchemaInteroperabilityAndEdgeCases:
    """Test schema interactions and edge cases comprehensively."""
    
    @pytest.mark.parametrize("calc_type,inputs,expected_result", [
        ("addition", [3.5, 2.0], 5.5),
        ("subtraction", [10, 3], 7.0),
        ("multiplication", [3.5, 2.0], 7.0),
        ("division", [10, 2], 5.0)
    ])
    def test_create_to_response_flow_individual_operations(self, calc_type: str, inputs: List[float], expected_result: float, sample_user_id, sample_datetime):
        """Test data flow from create to response for each operation individually."""
        # Create a calculation request
        create_data = {
            "type": calc_type,
            "inputs": inputs,
            "user_id": str(sample_user_id)
        }
        create_calc = CalculationCreate(**create_data)
        
        # Simulate application processing
        calc_id = uuid.uuid4()
        response_data = {
            "id": str(calc_id),
            "user_id": str(create_calc.user_id),
            "type": create_calc.type,
            "inputs": create_calc.inputs,
            "result": expected_result,
            "created_at": sample_datetime.isoformat(),
            "updated_at": sample_datetime.isoformat()
        }
        response_calc = CalculationResponse(**response_data)
        
        # Verify data consistency across schemas
        assert response_calc.user_id == create_calc.user_id
        assert response_calc.type == create_calc.type
        assert response_calc.inputs == create_calc.inputs
        assert response_calc.result == expected_result
    
    def test_edge_cases_comprehensive(self, sample_user_id):
        """Test edge cases including extreme numbers, precision, and boundaries."""
        edge_case_data = [
            ([1e308, 1.0], "extremely large numbers"),           # Large numbers
            ([1e-300, 2.0], "extremely small numbers"),          # Small numbers
            ([1.123456789012345, 2.987654321098765], "precision"),  # Precision
            (list(range(2, 102)), "large input list"),           # Large list (100 items)
            ([2, 2], "minimum valid length")                     # Boundary case
        ]
        
        for inputs, description in edge_case_data:
            data = {
                "type": "addition",
                "inputs": inputs,
                "user_id": str(sample_user_id)
            }
            calc = CalculationCreate(**data)
            assert len(calc.inputs) == len(inputs), f"Failed for {description}"
            if description == "precision":
                assert len(str(calc.inputs[0])) >= 10, "Precision not maintained"
    
    def test_unicode_and_error_handling(self):
        """Test unicode handling in error messages and validation."""
        # Test unicode symbols in invalid inputs
        data = {"type": "addition", "inputs": ["∞", "π"]}  # Unicode symbols
        
        with pytest.raises(ValidationError) as exc_info:
            CalculationBase(**data)
        
        # Should not raise encoding errors
        error_str = str(exc_info.value)
        assert isinstance(error_str, str)
    
    def test_update_schema_scenarios(self):
        """Test various update scenarios comprehensively."""
        # Test valid update with inputs
        update_calc = CalculationUpdate(inputs=[100, 25, 4])
        assert update_calc.inputs == [100, 25, 4]
        
        # Test empty update (no changes)
        empty_update = CalculationUpdate()
        assert empty_update.inputs is None
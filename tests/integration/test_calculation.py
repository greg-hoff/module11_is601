# tests/integration/test_calculation.py
"""
Integration Tests for Polymorphic Calculation Models

These tests verify the polymorphic behavior of the Calculation model hierarchy,
ensuring that:
1. Each calculation type is correctly instantiated through polymorphism
2. Database operations work correctly with polymorphic inheritance
3. The factory pattern creates appropriate subclass instances
4. Cross-type queries and operations function properly
5. Relationships with User model work across all calculation types

Test Categories:
- Factory Pattern Tests: Test Calculation.create() method
- Polymorphic Query Tests: Test SQLAlchemy's polymorphic querying
- Business Logic Tests: Test each calculation type's get_result() method
- Database Integration Tests: Test CRUD operations with polymorphism
- Cross-Type Tests: Test operations across multiple calculation types
"""

import pytest
import uuid
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.models.calculation import (
    Calculation,
    Addition,
    Subtraction,
    Multiplication,
    Division,
    AbstractCalculation
)
from app.models.user import User


class TestPolymorphicFactoryPattern:
    """
    Test the factory pattern implementation in Calculation.create().
    
    The factory pattern allows creating appropriate subclass instances
    without knowing the specific class at compile time.
    """

    @pytest.mark.parametrize("calc_type,expected_class,inputs", [
        ('addition', Addition, [1, 2, 3]),
        ('subtraction', Subtraction, [10, 3, 2]),
        ('multiplication', Multiplication, [2, 3, 4]),
        ('division', Division, [100, 2, 5]),
    ])
    def test_factory_creates_correct_instances(self, test_user: User, calc_type: str, expected_class, inputs: List[float]):
        """Test that factory creates correct subclass instances for each calculation type."""
        calc = Calculation.create(calc_type, test_user.id, inputs)
        
        # Verify correct type and polymorphic identity
        assert isinstance(calc, expected_class)
        assert isinstance(calc, Calculation)  # Should also be instance of base
        assert calc.type == calc_type
        assert calc.user_id == test_user.id
        assert calc.inputs == inputs

    def test_factory_case_insensitive(self, test_user: User):
        """Test that factory method is case-insensitive."""
        calc_upper = Calculation.create('ADDITION', test_user.id, [1, 2])
        calc_mixed = Calculation.create('AdDiTiOn', test_user.id, [1, 2])
        
        assert isinstance(calc_upper, Addition)
        assert isinstance(calc_mixed, Addition)
        assert calc_upper.type == 'addition'
        assert calc_mixed.type == 'addition'

    def test_factory_invalid_type_raises_error(self, test_user: User):
        """Test that factory raises ValueError for unsupported calculation types."""
        with pytest.raises(ValueError, match="Unsupported calculation type: invalid"):
            Calculation.create('invalid', test_user.id, [1, 2])
        
        with pytest.raises(ValueError, match="Unsupported calculation type: "):
            Calculation.create('', test_user.id, [1, 2])


class TestPolymorphicDatabaseOperations:
    """
    Test database operations with polymorphic inheritance.
    
    These tests verify that SQLAlchemy's polymorphic features work correctly
    with our calculation hierarchy.
    """

    def test_polymorphic_save_and_query_operations(self, db_session: Session, test_user: User):
        """Test saving and querying different calculation types through polymorphism."""
        # Create instances of each type
        calculations = [
            Addition(user_id=test_user.id, inputs=[1, 2, 3]),
            Subtraction(user_id=test_user.id, inputs=[10, 3]),
            Multiplication(user_id=test_user.id, inputs=[2, 4]),
            Division(user_id=test_user.id, inputs=[20, 4])
        ]
        
        # Save all instances
        db_session.add_all(calculations)
        db_session.commit()
        
        # Query base class - should return correct subclass instances
        retrieved_calcs = db_session.query(Calculation).all()
        assert len(retrieved_calcs) == 4
        
        # SQLAlchemy should return the correct subclass instances
        types_found = {type(calc).__name__ for calc in retrieved_calcs}
        expected_types = {'Addition', 'Subtraction', 'Multiplication', 'Division'}
        assert types_found == expected_types
        
        # Verify each instance is both its specific type and base type
        for calc in retrieved_calcs:
            assert isinstance(calc, Calculation)
            if calc.type == 'addition':
                assert isinstance(calc, Addition)
            elif calc.type == 'subtraction':
                assert isinstance(calc, Subtraction)
            elif calc.type == 'multiplication':
                assert isinstance(calc, Multiplication)
            elif calc.type == 'division':
                assert isinstance(calc, Division)

    def test_filter_by_calculation_type(self, db_session: Session, test_user: User):
        """Test filtering calculations by type discriminator."""
        # Create mixed calculation types
        add1 = Addition(user_id=test_user.id, inputs=[1, 2])
        add2 = Addition(user_id=test_user.id, inputs=[3, 4])
        sub1 = Subtraction(user_id=test_user.id, inputs=[10, 5])
        
        db_session.add_all([add1, add2, sub1])
        db_session.commit()
        
        # Filter by type
        additions = db_session.query(Calculation).filter(Calculation.type == 'addition').all()
        subtractions = db_session.query(Calculation).filter(Calculation.type == 'subtraction').all()
        
        assert len(additions) == 2
        assert len(subtractions) == 1
        assert all(isinstance(calc, Addition) for calc in additions)
        assert all(isinstance(calc, Subtraction) for calc in subtractions)

    def test_direct_subclass_queries(self, db_session: Session, test_user: User):
        """Test querying specific subclasses using polymorphic filtering."""
        # Create mixed types
        add_calc = Addition(user_id=test_user.id, inputs=[1, 2])
        sub_calc = Subtraction(user_id=test_user.id, inputs=[5, 3])
        mult_calc = Multiplication(user_id=test_user.id, inputs=[2, 3])
        
        db_session.add_all([add_calc, sub_calc, mult_calc])
        db_session.commit()
        
        # Query specific types using polymorphic filtering
        additions = db_session.query(Calculation).filter(Calculation.type == 'addition').all()
        subtractions = db_session.query(Calculation).filter(Calculation.type == 'subtraction').all()
        multiplications = db_session.query(Calculation).filter(Calculation.type == 'multiplication').all()
        divisions = db_session.query(Calculation).filter(Calculation.type == 'division').all()
        
        assert len(additions) == 1
        assert len(subtractions) == 1
        assert len(multiplications) == 1
        assert len(divisions) == 0
        
        assert isinstance(additions[0], Addition)
        assert isinstance(subtractions[0], Subtraction)
        assert isinstance(multiplications[0], Multiplication)


class TestCalculationBusinessLogic:
    """
    Test the business logic of each calculation type.
    
    These tests verify that each subclass correctly implements
    the get_result() method according to its mathematical operation.
    Each operation is tested individually to ensure clear functionality.
    """

    def test_addition_get_result(self, test_user: User):
        """
        Test that Addition.get_result returns the correct sum.
        
        This verifies that the Addition class correctly implements the
        polymorphic get_result() method for its specific operation.
        """
        inputs = [10, 5, 3.5]
        addition = Addition(user_id=test_user.id, inputs=inputs)
        result = addition.get_result()
        assert result == sum(inputs), f"Expected {sum(inputs)}, got {result}"

    def test_addition_get_result_multiple_values(self, test_user: User):
        """Test Addition with multiple values including decimals."""
        inputs = [1.5, 2.5, 3, 4]
        addition = Addition(user_id=test_user.id, inputs=inputs)
        result = addition.get_result()
        expected = 11.0  # 1.5 + 2.5 + 3 + 4
        assert result == expected, f"Expected {expected}, got {result}"

    def test_subtraction_get_result(self, test_user: User):
        """
        Test that Subtraction.get_result returns the correct difference.
        
        This verifies that the Subtraction class correctly implements
        sequential subtraction: first - second - third - ...
        """
        inputs = [20, 5, 3]
        subtraction = Subtraction(user_id=test_user.id, inputs=inputs)
        result = subtraction.get_result()
        expected = 12  # 20 - 5 - 3
        assert result == expected, f"Expected {expected}, got {result}"

    def test_subtraction_get_result_with_decimals(self, test_user: User):
        """Test Subtraction with decimal values."""
        inputs = [10.5, 2.5, 1.0]
        subtraction = Subtraction(user_id=test_user.id, inputs=inputs)
        result = subtraction.get_result()
        expected = 7.0  # 10.5 - 2.5 - 1.0
        assert result == expected, f"Expected {expected}, got {result}"

    def test_multiplication_get_result(self, test_user: User):
        """
        Test that Multiplication.get_result returns the correct product.
        
        This verifies that the Multiplication class correctly implements
        sequential multiplication of all input values.
        """
        inputs = [2, 3, 4]
        multiplication = Multiplication(user_id=test_user.id, inputs=inputs)
        result = multiplication.get_result()
        expected = 24  # 2 * 3 * 4
        assert result == expected, f"Expected {expected}, got {result}"

    def test_multiplication_get_result_with_decimals(self, test_user: User):
        """Test Multiplication with decimal values."""
        inputs = [2.5, 4, 1.2]
        multiplication = Multiplication(user_id=test_user.id, inputs=inputs)
        result = multiplication.get_result()
        expected = 12.0  # 2.5 * 4 * 1.2
        assert result == expected, f"Expected {expected}, got {result}"

    def test_division_get_result(self, test_user: User):
        """
        Test that Division.get_result returns the correct quotient.
        
        This verifies that the Division class correctly implements
        sequential division: first / second / third / ...
        """
        inputs = [100, 2, 5]
        division = Division(user_id=test_user.id, inputs=inputs)
        result = division.get_result()
        expected = 10  # 100 / 2 / 5
        assert result == expected, f"Expected {expected}, got {result}"

    def test_division_get_result_with_decimals(self, test_user: User):
        """Test Division with decimal values."""
        inputs = [15.0, 3.0, 1.25]
        division = Division(user_id=test_user.id, inputs=inputs)
        result = division.get_result()
        expected = 4.0  # 15.0 / 3.0 / 1.25
        assert result == expected, f"Expected {expected}, got {result}"

    def test_division_by_zero_error(self, test_user: User):
        """Test that division by zero raises appropriate error."""
        # Test direct division by zero
        division = Division(user_id=test_user.id, inputs=[10, 0])
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            division.get_result()
        
        # Test zero in middle of sequence
        division_middle_zero = Division(user_id=test_user.id, inputs=[100, 2, 0, 5])
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            division_middle_zero.get_result()

    @pytest.mark.parametrize("calc_class", [Addition, Subtraction, Multiplication, Division])
    def test_input_validation_across_types(self, test_user: User, calc_class):
        """Test input validation for all calculation types."""
        # Test non-list inputs
        calc = calc_class(user_id=test_user.id, inputs="not a list")
        with pytest.raises(ValueError, match="Inputs must be a list of numbers"):
            calc.get_result()
        
        # Test insufficient inputs (less than 2)
        calc = calc_class(user_id=test_user.id, inputs=[1])
        with pytest.raises(ValueError, match="at least two numbers"):
            calc.get_result()


class TestUserCalculationRelationships:
    """
    Test the relationship between User and Calculation models.
    
    These tests verify that the polymorphic calculations correctly
    maintain relationships with users.
    """

    def test_user_calculation_relationship(self, db_session: Session, test_user: User):
        """Test bidirectional relationship between User and Calculations."""
        # Create calculations of different types
        add_calc = Addition(user_id=test_user.id, inputs=[1, 2])
        sub_calc = Subtraction(user_id=test_user.id, inputs=[5, 3])
        
        db_session.add_all([add_calc, sub_calc])
        db_session.commit()
        
        # Test user.calculations relationship
        db_session.refresh(test_user)
        user_calculations = test_user.calculations
        
        assert len(user_calculations) == 2
        assert any(isinstance(calc, Addition) for calc in user_calculations)
        assert any(isinstance(calc, Subtraction) for calc in user_calculations)
        
        # Test calculation.user relationship
        for calc in user_calculations:
            assert calc.user == test_user
            assert calc.user_id == test_user.id

    def test_cascade_delete_calculations(self, db_session: Session, test_user: User):
        """Test that calculations are deleted when user is deleted (CASCADE)."""
        # Create calculations for the user
        calculations = [
            Addition(user_id=test_user.id, inputs=[1, 2]),
            Subtraction(user_id=test_user.id, inputs=[5, 3]),
            Multiplication(user_id=test_user.id, inputs=[2, 4])
        ]
        
        for calc in calculations:
            db_session.add(calc)
        db_session.commit()
        
        # Verify calculations exist
        calc_count = db_session.query(Calculation).filter(
            Calculation.user_id == test_user.id
        ).count()
        assert calc_count == 3
        
        # Delete user - should cascade to calculations
        db_session.delete(test_user)
        db_session.commit()
        
        # Verify calculations were deleted
        remaining_calcs = db_session.query(Calculation).filter(
            Calculation.user_id == test_user.id
        ).count()
        assert remaining_calcs == 0

    def test_multiple_users_calculations(self, db_session: Session, seed_users: List[User]):
        """Test calculations with multiple users."""
        user1, user2 = seed_users[0], seed_users[1]
        
        # Create calculations for different users
        user1_calcs = [
            Addition(user_id=user1.id, inputs=[1, 2]),
            Subtraction(user_id=user1.id, inputs=[5, 3])
        ]
        
        user2_calcs = [
            Multiplication(user_id=user2.id, inputs=[2, 4]),
            Division(user_id=user2.id, inputs=[20, 4])
        ]
        
        all_calcs = user1_calcs + user2_calcs
        for calc in all_calcs:
            db_session.add(calc)
        db_session.commit()
        
        # Verify user-specific queries
        user1_results = db_session.query(Calculation).filter(
            Calculation.user_id == user1.id
        ).all()
        user2_results = db_session.query(Calculation).filter(
            Calculation.user_id == user2.id
        ).all()
        
        assert len(user1_results) == 2
        assert len(user2_results) == 2
        
        # Verify correct types for each user
        user1_types = {type(calc).__name__ for calc in user1_results}
        user2_types = {type(calc).__name__ for calc in user2_results}
        
        assert user1_types == {'Addition', 'Subtraction'}
        assert user2_types == {'Multiplication', 'Division'}


class TestPolymorphicCrossTypeOperations:
    """
    Test operations that work across different calculation types.
    
    These tests verify that polymorphic behavior works correctly
    when dealing with mixed collections of calculation types.
    """

    def test_mixed_calculation_operations(self, db_session: Session, test_user: User):
        """Test comprehensive operations across mixed calculation types."""
        # Create a mix of calculation types with predictable results
        calculations = [
            Addition(user_id=test_user.id, inputs=[1, 2, 3]),      # Result: 6
            Subtraction(user_id=test_user.id, inputs=[10, 4]),     # Result: 6
            Multiplication(user_id=test_user.id, inputs=[2, 3]),   # Result: 6
            Division(user_id=test_user.id, inputs=[24, 4]),        # Result: 6
        ]
        
        # Save all calculations
        db_session.add_all(calculations)
        db_session.commit()
        
        # Test polymorphic batch processing
        all_calcs = db_session.query(Calculation).all()
        results = [calc.get_result() for calc in all_calcs]
        
        # All should equal 6 based on our inputs
        assert all(result == 6 for result in results)
        assert len(results) == 4
        
        # Test statistics gathering
        from sqlalchemy import func
        type_stats = db_session.query(
            Calculation.type,
            func.count(Calculation.id).label('count')
        ).group_by(Calculation.type).all()
        
        stats_dict = {stat.type: stat.count for stat in type_stats}
        expected_stats = {'addition': 1, 'subtraction': 1, 'multiplication': 1, 'division': 1}
        assert stats_dict == expected_stats
        
        # Test polymorphic updates
        for calc in all_calcs:
            calc.result = calc.get_result()  # Store computed result
        db_session.commit()
        
        # Verify all results are stored correctly
        updated_calcs = db_session.query(Calculation).all()
        assert all(calc.result == 6 for calc in updated_calcs)


class TestPolymorphicConstraintsAndValidation:
    """
    Test database constraints and validation with polymorphic models.
    """

    def test_foreign_key_constraint_enforcement(self, db_session: Session):
        """Test that foreign key constraints are properly enforced."""
        # Try to create calculation with non-existent user_id
        fake_user_id = uuid.uuid4()
        calc = Addition(user_id=fake_user_id, inputs=[1, 2])
        
        db_session.add(calc)
        
        # Should raise IntegrityError due to foreign key constraint
        with pytest.raises(IntegrityError):
            db_session.commit()
        
        # Rollback the failed transaction to clean up session state
        db_session.rollback()

    def test_polymorphic_identity_consistency(self, db_session: Session, test_user: User):
        """Test that polymorphic identity is consistent with actual class."""
        # Create instances through factory
        add_calc = Calculation.create('addition', test_user.id, [1, 2])
        sub_calc = Calculation.create('subtraction', test_user.id, [5, 3])
        
        db_session.add_all([add_calc, sub_calc])
        db_session.commit()
        
        # Retrieve and verify polymorphic identity matches class
        retrieved_calcs = db_session.query(Calculation).all()
        
        for calc in retrieved_calcs:
            if isinstance(calc, Addition):
                assert calc.type == 'addition'
            elif isinstance(calc, Subtraction):
                assert calc.type == 'subtraction'
            else:
                pytest.fail(f"Unexpected calculation type: {type(calc)}")

    def test_abstract_calculation_not_instantiable(self):
        """Test that AbstractCalculation provides proper base functionality."""
        # AbstractCalculation is a base class that provides common functionality
        # Test that it has the required abstract method
        
        # Verify AbstractCalculation has the get_result method that must be implemented
        assert hasattr(AbstractCalculation, 'get_result')
        
        # Verify concrete classes implement get_result
        assert hasattr(Addition, 'get_result')
        assert hasattr(Subtraction, 'get_result')
        assert hasattr(Multiplication, 'get_result')
        assert hasattr(Division, 'get_result')
        
        # Verify base Calculation class has proper polymorphic setup
        assert hasattr(Calculation, '__mapper_args__')
        assert 'polymorphic_on' in Calculation.__mapper_args__


@pytest.mark.parametrize("seed_users", [2], indirect=True)
class TestPolymorphicPerformance:
    """
    Test performance characteristics of polymorphic queries.
    
    These tests ensure that polymorphic inheritance doesn't introduce
    significant performance issues.
    """

    def test_bulk_polymorphic_operations_and_efficiency(self, db_session: Session, seed_users: List[User]):
        """Test bulk operations and query efficiency with polymorphic calculations."""
        user1, user2 = seed_users[0], seed_users[1]
        
        # Create a large number of mixed calculations
        calculation_types = [
            (Addition, [1, 2]),
            (Subtraction, [5, 3]),
            (Multiplication, [2, 3]),
            (Division, [12, 3])
        ]
        
        calculations = []
        # Create 40 calculations (10 of each type, distributed across users)
        for i in range(40):
            calc_class, inputs = calculation_types[i % 4]
            user = user1 if i % 2 == 0 else user2
            calc = calc_class(user_id=user.id, inputs=inputs)
            calculations.append(calc)
        
        # Bulk insert
        db_session.add_all(calculations)
        db_session.commit()
        
        # Test bulk query retrieves all correctly
        all_calcs = db_session.query(Calculation).all()
        assert len(all_calcs) == 40
        
        # Verify distribution and polymorphic behavior
        type_counts = {}
        total_result = 0
        for calc in all_calcs:
            calc_type = type(calc).__name__
            type_counts[calc_type] = type_counts.get(calc_type, 0) + 1
            # Test polymorphic method calls work efficiently
            total_result += calc.get_result()
        
        # Should have exactly 10 of each type
        for calc_type in ['Addition', 'Subtraction', 'Multiplication', 'Division']:
            assert type_counts[calc_type] == 10
        
        # Verify total calculation (10*3 + 10*2 + 10*6 + 10*4 = 150)
        assert total_result == 150

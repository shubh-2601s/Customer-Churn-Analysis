import mysql.connector
import pandas as pd

# MySQL connection details - replace with your own
config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Shubh_26',
    'database': 'customerchurn',
    'port': 3306  # change if needed
}

try:
    # Connect to the database
    conn = mysql.connector.connect(**config)

    # Define your query
    query = """
    SELECT
      customer_id,
      age,
      gender,
      region,
      plan_type,
      monthly_charges,
      total_recharges,
      last_recharge_date,
      calls_made,
      call_duration_total,
      num_complaints,
      internet_usage_GB,
      support_tickets,
      contract_type,
      CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END AS churn_flag
    FROM telecom_customers;
    """

    # Load data into DataFrame
    df = pd.read_sql(query, conn)

    # Export to CSV (change the path as needed)
    output_path = 'telecom_customers.csv'
    df.to_csv(output_path, index=False)
    print(f"Data exported successfully to {output_path}")

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    if 'conn' in locals() and conn.is_connected():
        conn.close()

# ==============================================================================
# OLIST ABT LOADING SCRIPT (abttable.py) - COMPREHENSIVE VERSION
# ==============================================================================
# Purpose: Connects to the Olist MySQL database, executes a query to
#          build a richer Analytical Base Table (ABT) suitable for various
#          analyses including identifiers and more detailed features.
#          Loads data efficiently using chunking and saves the
#          resulting DataFrame to a file (Parquet preferred, CSV fallback).
# Output:  A file named 'olist_abt_loaded_comprehensive.parquet' (or .csv)
#          containing the comprehensive ABT.
# ==============================================================================

import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import time # To time the loading process

# --- Configuration ---
# Database Credentials
DB_HOST = "localhost"
DB_USER = "teena"
DB_PASSWORD = "Teena@123#Pass" # Consider using env variables or config file
DB_NAME = "olist_db"

# Loading Parameters
CHUNK_SIZE = 10000 # Adjust based on your system's RAM

# Output Filenames (Using distinct names for this comprehensive version)
OUTPUT_FILENAME_PARQUET = 'olist_abt_loaded_comprehensive.parquet'
OUTPUT_FILENAME_CSV = 'olist_abt_loaded_comprehensive.csv'

print("--- Olist Comprehensive ABT Loading Script ---")
print(f"Target Database: {DB_NAME} on {DB_HOST}")
print(f"Chunk Size for Loading: {CHUNK_SIZE}")
print(f"Output (Parquet): {OUTPUT_FILENAME_PARQUET}")
print(f"Output (CSV Fallback): {OUTPUT_FILENAME_CSV}")
print("------------------------------------------\n")

# --- Initialize Variables ---
df_abt = None
engine = None
start_time_script = time.time()

try:
    # --- 1. Create Database Engine ---
    print("Connecting to database...")
    encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
    connection_string = f'mysql+mysqlconnector://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}'
    # echo=False prevents printing all SQL commands executed by SQLAlchemy
    # Add connect_args={'connect_timeout': 10} within create_engine if needed
    engine = create_engine(connection_string, echo=False)
    print("Database engine created.")

    # --- 2. Define the Comprehensive ABT SQL Query ---
    # Includes identifiers, raw timestamps, calculated durations, city, English category, etc.
    print("Defining Comprehensive ABT SQL query...")
    abt_query = """
    WITH PaymentInfo AS (
        -- Aggregate payment information per order
        SELECT
            order_id,
            COUNT(payment_sequential) AS payment_count, -- Number of payment steps/methods
            MAX(payment_type) AS payment_type, -- Simple aggregation for type
            MAX(payment_installments) AS payment_installments, -- Max installments used
            SUM(payment_value) AS payment_value -- Total payment value
        FROM order_payments
        GROUP BY order_id
    ),
    ItemProductInfo AS (
        -- Aggregate item and associated product information per order
        SELECT
            oi.order_id,
            SUM(oi.price) AS total_price,
            SUM(oi.freight_value) AS total_freight,
            COUNT(oi.order_item_id) AS num_items,
            COUNT(DISTINCT oi.seller_id) AS distinct_sellers,
            AVG(p.product_weight_g) AS avg_product_weight_g,
            -- Calculate volume, handle potential NULL dimensions using COALESCE
            AVG( COALESCE(p.product_length_cm, 0) *
                 COALESCE(p.product_height_cm, 0) *
                 COALESCE(p.product_width_cm, 0) ) AS avg_product_volume_cm3,
            AVG(p.product_name_lenght) AS avg_product_name_length,
            AVG(p.product_description_lenght) AS avg_product_desc_length,
            AVG(p.product_photos_qty) AS avg_product_photos_qty,
            -- Get English category name, fallback to original, then 'Unknown'
            -- Ensure product_category_name_translation table exists and is joined correctly
            COALESCE(MAX(pct.product_category_name_english), MAX(p.product_category_name), 'Unknown') AS product_category_name_english
        FROM order_items oi
        LEFT JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN product_category_name_translation pct ON p.product_category_name = pct.product_category_name
        GROUP BY oi.order_id
    )
    -- Final SELECT joining all components
    SELECT
        -- === Identifiers (Keep for lookup, EXCLUDE from direct ML features) ===
        o.order_id,
        r.review_id,
        c.customer_id,
        c.customer_unique_id, -- Tracks the customer across multiple orders

        -- === Target Variable(s) ===
        r.review_score,
        CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END AS is_low_score, -- Binary target

        -- === Order Timing Features (Calculated Durations) ===
        TIMESTAMPDIFF(DAY, o.order_purchase_timestamp, o.order_delivered_customer_date) AS delivery_days,
        TIMESTAMPDIFF(DAY, o.order_estimated_delivery_date, o.order_delivered_customer_date) AS delivery_diff_days, -- Lateness/Earliness
        TIMESTAMPDIFF(HOUR, o.order_purchase_timestamp, o.order_approved_at) AS approval_hours,
        TIMESTAMPDIFF(HOUR, o.order_approved_at, o.order_delivered_carrier_date) AS processing_hours,
        TIMESTAMPDIFF(HOUR, o.order_delivered_carrier_date, o.order_delivered_customer_date) AS carrier_hours,

        -- === Review Timing Features ===
        TIMESTAMPDIFF(HOUR, r.review_creation_date, r.review_answer_timestamp) AS review_response_hours, -- Review response time

        -- === Raw Timestamp Features (Allows further engineering in Python) ===
        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_carrier_date,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,
        r.review_creation_date,
        r.review_answer_timestamp,

        -- === Customer Features ===
        c.customer_city, -- High Cardinality - handle with care later
        c.customer_state,
        c.customer_zip_code_prefix, -- Potentially high cardinality

        -- === Payment Features (from CTE) ===
        pay.payment_type,
        pay.payment_installments,
        pay.payment_value,
        pay.payment_count,

        -- === Aggregated Item/Product Features (from CTE) ===
        ipi.total_price,
        ipi.total_freight,
        ipi.num_items,
        ipi.distinct_sellers,
        ipi.avg_product_weight_g,
        ipi.avg_product_volume_cm3,
        ipi.avg_product_name_length,
        ipi.avg_product_desc_length,
        ipi.avg_product_photos_qty,
        ipi.product_category_name_english -- Use the translated name

    FROM order_reviews r
    JOIN orders o ON r.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN PaymentInfo pay ON o.order_id = pay.order_id
    LEFT JOIN ItemProductInfo ipi ON o.order_id = ipi.order_id

    -- Filter for relevant orders: Must be delivered and have key timestamps populated
    -- Stricter timestamp checks needed for all calculated durations
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
      AND o.order_approved_at IS NOT NULL
      AND o.order_delivered_carrier_date IS NOT NULL
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
      AND r.review_creation_date IS NOT NULL
      AND r.review_answer_timestamp IS NOT NULL;
    """

    # --- 3. Load Data using Chunking with Explicit Connection ---
    print("\nStarting data loading from database using chunks...")
    start_time_load = time.time()
    chunk_list = []
    total_rows = 0

    # Use context manager for the connection to help prevent sync errors
    with engine.connect() as connection:
        print("  Acquired database connection.")
        # Pass the specific connection, not the engine, to read_sql
        sql_iterator = pd.read_sql(abt_query, connection, chunksize=CHUNK_SIZE)

        for i, chunk in enumerate(sql_iterator):
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            # Print progress less frequently for faster execution, e.g., every 10 chunks
            if (i + 1) % 10 == 0 or chunk_rows < CHUNK_SIZE:
                 print(f"    Processed chunk {i+1} ({chunk_rows} rows). Total rows so far: {total_rows}")
            chunk_list.append(chunk)
        # Connection automatically closed/returned by context manager here
        print("  Finished iterating through chunks. Connection closed/returned.")

    end_time_load = time.time()
    print(f"\nFinished reading {total_rows} rows from database in {end_time_load - start_time_load:.2f} seconds.")

    # --- 4. Concatenate Chunks ---
    if not chunk_list:
        print("Warning: No data was loaded. Check your SQL query and filters.")
        df_abt = pd.DataFrame() # Create empty DataFrame
    else:
        print("Attempting to concatenate chunks...")
        start_concat_time = time.time()
        try:
            # Concatenate all loaded chunks into the final DataFrame
            df_abt = pd.concat(chunk_list, ignore_index=True)
            end_concat_time = time.time()
            print(f"Successfully concatenated chunks in {end_concat_time - start_concat_time:.2f} seconds.")
            print(f"Final DataFrame shape: {df_abt.shape}")
            # Display info about the loaded comprehensive table
            print("\nLoaded DataFrame Info Sample:")
            df_abt.info(verbose=True, show_counts=True) # Show more details if needed
        except MemoryError:
            print("\n********************************************************************")
            print("ERROR: MemoryError occurred while concatenating chunks.")
            print("The complete dataset is too large to fit into available RAM.")
            print("Try decreasing CHUNK_SIZE or using a machine with more RAM.")
            print("********************************************************************")
            df_abt = None # Indicate failure
        except Exception as concat_err:
             print(f"\nERROR during concat: {concat_err}")
             df_abt = None # Indicate failure

# --- Handle potential errors during connection or query ---
except Exception as e:
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"An critical error occurred during script execution:")
    print(e)
    print("Check database status, credentials, SQL syntax, table/column names, and permissions.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # df_abt remains None or its previous state

# --- 5. Save the Loaded DataFrame ---
finally:
    # Save the result if loading/concatenation was successful
    if df_abt is not None and not df_abt.empty:
        print("\n--- Saving Loaded Comprehensive DataFrame ---")
        saved_successfully = False
        # Try Parquet first (Recommended)
        try:
            print(f"Attempting to save as Parquet: '{OUTPUT_FILENAME_PARQUET}'")
            # Ensure pyarrow is installed: pip install pyarrow
            df_abt.to_parquet(OUTPUT_FILENAME_PARQUET, index=False, engine='pyarrow') # Specify engine if needed
            print(f"Successfully saved comprehensive data to '{OUTPUT_FILENAME_PARQUET}'")
            saved_successfully = True
        except ImportError:
             print("  'pyarrow' library not found. Cannot save as Parquet.")
             print("  Install it using: pip install pyarrow")
        except Exception as e_parquet:
            print(f"  Error saving to Parquet: {e_parquet}")

        # Fallback to CSV if Parquet failed
        if not saved_successfully:
            try:
                print(f"\nAttempting to save as CSV: '{OUTPUT_FILENAME_CSV}'")
                df_abt.to_csv(OUTPUT_FILENAME_CSV, index=False)
                print(f"Successfully saved comprehensive data to '{OUTPUT_FILENAME_CSV}'")
                saved_successfully = True
            except Exception as e_csv:
                print(f"  Error saving to CSV: {e_csv}")

        if not saved_successfully:
             print("\nERROR: Failed to save the comprehensive DataFrame to any format.")

    elif df_abt is not None and df_abt.empty:
         print("\nSkipping save: Loaded DataFrame is empty.")
    else:
        print("\nSkipping save: DataFrame was not loaded correctly due to previous errors.")

    # --- 6. Dispose Database Engine ---
    # Ensures connection pool resources are released
    if engine:
        print("\nDisposing database engine pool...")
        engine.dispose()
        print("Database engine pool disposed.")

end_time_script = time.time()
print("\n--- Olist COMPREHENSIVE ABT Loading Script Finished ---")
print(f"Total script execution time: {end_time_script - start_time_script:.2f} seconds.")
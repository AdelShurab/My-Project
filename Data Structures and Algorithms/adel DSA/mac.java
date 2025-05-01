public class mac {
    public static void main(String[] args) {
        // Create new queue
        queue.Queue queue = new queue.Queue();

        // Create order 1
        DLLitem.DLLItems items1 = new DLLitem.DLLItems();
        items1.addItem(new item.Item("Big tasty", 5.5));
        items1.addItem(new item.Item("mccoffee", 3.0));
        order.Order order1 = new order.Order(1, items1);
        order1.calculateTotalPrice();

        // Add order 1 to queue
        queue.addOrder(order1);

        // Create order 2
        DLLitem.DLLItems items2 = new DLLitem.DLLItems();
        items2.addItem(new item.Item("mcflurry oreo", 3.5));
        items2.addItem(new item.Item("pepsi", 1.0));
        items2.addItem(new item.Item("chicken mcnuggets", 3.0));
        order.Order order2 = new order.Order(2, items2);
        order2.calculateTotalPrice();

        // Add order 2 to queue
        queue.addOrder(order2);

        // Serve the orders
        order.Order servedOrder;
        while((servedOrder = queue.removeOrder()) != null) {
            System.out.println("Order id " + servedOrder.getOrderID());
            System.out.println("Order items :");
            servedOrder.printOrder();
            System.out.println("***");
        }
        queue.Queue myQueue = new queue.Queue();
        if (myQueue.isEmpty()) {
            System.out.println("The queue is empty");
        } else {
            System.out.println("The queue is not empty");
        }

        System.out.println("queue is empty");
    }
}